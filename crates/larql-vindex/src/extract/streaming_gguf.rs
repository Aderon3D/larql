//! Streaming vindex extraction for GGUF — build without loading the full model.
//!
//! Peak memory: embeddings + 1 layer of gate/down weights at a time.
//! Processes GGUF files by dequantizing tensors on-demand.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array2;
use larql_models::loading::gguf::{GgufFile, normalize_gguf_key};
use larql_models::quant::ggml::dequantize;

use crate::config::dtype::StorageDtype;
use crate::config::{VindexConfig, VindexLayerInfo, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;

/// Build a vindex by streaming from a GGUF file.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex_streaming_gguf(
    model_path: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    std::fs::create_dir_all(output_dir)?;

    // Open GGUF file and parse metadata
    let gguf = GgufFile::open(model_path)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    
    // Detect architecture
    let config_json = gguf.to_config_json();
    let arch = larql_models::detect_from_json(&config_json);
    let prefixes = arch.key_prefixes_to_strip();
    let cfg = arch.config();

    let num_layers = cfg.num_layers;
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg.intermediate_size;
    let embed_scale = arch.embed_scale();
    let is_moe = arch.is_moe();
    let n_experts = arch.num_experts();

    // Mmap the GGUF file for fast tensor access
    let file = std::fs::File::open(model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // Build tensor index: normalized_key -> &GgufTensorInfo
    let mut tensor_index = HashMap::new();
    for info in &gguf.tensor_infos {
        let key = normalize_key(&info.name, prefixes);
        tensor_index.insert(key, info);
    }

    callbacks.on_stage("gate_vectors");
    let gate_path = output_dir.join("gate_vectors.bin");
    let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
    let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
    let mut offset: u64 = 0;

    // ── 1. Gate vectors (streaming) ──
    for layer in 0..num_layers {
        callbacks.on_layer_start("gate", layer, num_layers);
        let start = std::time::Instant::now();

        // Dense only for now in GGUF (MoE GGUFs exist but we prioritize dense E4B)
        let gate_key = normalize_key(&arch.ffn_gate_key(layer), prefixes);
        if let Some(info) = tensor_index.get(&gate_key) {
            let tensor = get_tensor_f32(&mmap, gguf.data_offset, info)?;
            let num_features = tensor.shape()[0];
            let data = tensor.as_slice().unwrap();
            let length = write_floats(&mut gate_file, data, dtype)?;
            layer_infos.push(VindexLayerInfo {
                layer, num_features, offset, length,
                num_experts: None, num_features_per_expert: None,
            });
            offset += length;
        }

        callbacks.on_layer_done("gate", layer, start.elapsed().as_secs_f64() * 1000.0);
    }
    gate_file.flush()?;
    callbacks.on_stage_done("gate_vectors", 0.0);

    // ── 2. Embeddings ──
    callbacks.on_stage("embeddings");
    let embed_key = normalize_key(arch.embed_key(), prefixes);
    let embed_info = tensor_index.get(&embed_key)
        .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
    let embed = get_tensor_f32(&mmap, gguf.data_offset, embed_info)?;
    
    // GGUF stores embeddings as [hidden_size, vocab_size], we need [vocab_size, hidden_size]
    let embed = if embed.shape()[0] < embed.shape()[1] {
        let mut out = Array2::<f32>::zeros((embed.shape()[1], embed.shape()[0]));
        out.assign(&embed.t());
        out
    } else {
        embed
    };
    
    let vocab_size = embed.shape()[0];
    let embed_data = embed.as_slice().unwrap();
    let embed_bytes = crate::config::dtype::encode_floats(embed_data, dtype);
    std::fs::write(output_dir.join("embeddings.bin"), &embed_bytes)?;
    callbacks.on_stage_done("embeddings", 0.0);

    // ── 3. Down meta (streaming) ──
    callbacks.on_stage("down_meta");
    let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> = vec![None; num_layers];

    for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(num_layers) {
        callbacks.on_layer_start("down", layer, num_layers);
        let start = std::time::Instant::now();

        let down_key = normalize_key(&arch.ffn_down_key(layer), prefixes);
        if let Some(info) = tensor_index.get(&down_key) {
            let w_down = get_tensor_f32(&mmap, gguf.data_offset, info)?;
            let num_features = w_down.shape()[1];
            let batch_size = 1024;

            for batch_start in (0..num_features).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_features);
                callbacks.on_feature_progress("down", layer, batch_start, num_features);

                let w_chunk = w_down.slice(ndarray::s![.., batch_start..batch_end]).to_owned();
                let cpu = larql_compute::CpuBackend;
                use larql_compute::ComputeBackend;
                let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());

                for feat in batch_start..batch_end {
                    let col = chunk_logits.column(feat - batch_start);
                    let mut scores: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();
                    let k = down_top_k.min(scores.len());
                    if k > 0 && k < scores.len() {
                        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    }
                    scores.truncate(k);
                    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    let top_k_entries: Vec<larql_models::TopKEntry> = scores.into_iter()
                        .filter_map(|(idx, logit)| {
                            tokenizer.decode(&[idx as u32], true).ok()
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .map(|token| larql_models::TopKEntry { token, token_id: idx as u32, logit })
                        })
                        .collect();

                    let (top_token, top_token_id, c_score) = if let Some(first) = top_k_entries.first() {
                        (first.token.clone(), first.token_id, first.logit)
                    } else {
                        (String::new(), 0, 0.0)
                    };

                    if layer_down_meta.is_none() {
                        *layer_down_meta = Some(Vec::new());
                    }
                    if let Some(ref mut metas) = layer_down_meta {
                        while metas.len() <= feat { metas.push(None); }
                        metas[feat] = Some(crate::FeatureMeta {
                            top_token, top_token_id, c_score, top_k: top_k_entries,
                        });
                    }
                }
            }
        }

        callbacks.on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);
    }

    crate::format::down_meta::write_binary(output_dir, &all_down_meta, down_top_k)?;
    callbacks.on_stage_done("down_meta", 0.0);

    // ── 4. Tokenizer ──
    callbacks.on_stage("tokenizer");
    let tokenizer_json = tokenizer.to_string(true)
        .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
    std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
    callbacks.on_stage_done("tokenizer", 0.0);

    // ── 5. Config ──
    let family = arch.family().to_string();
    let config = VindexConfig {
        version: 2,
        model: model_name.to_string(),
        family: family.clone(),
        num_layers, hidden_size, intermediate_size, vocab_size,
        embed_scale,
        layers: layer_infos,
        down_top_k,
        has_model_weights: false,
        source: Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: super::build::chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        checksums: None,
        extract_level,
        dtype,
        layer_bands: crate::LayerBands::for_family(&family, num_layers),
        model_config: Some(VindexModelConfig {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if is_moe {
                Some(crate::MoeConfig {
                    num_experts: n_experts,
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: "top_k_softmax".to_string(),
                })
            } else { None },
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
        }),
    };

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    // ── 6. Model weights (Optional) ──
    // For now, GGUF model weights extraction (up/down/attn) is not yet implemented in streaming mode
    // because it requires a WeightsSource that handles GGUF.
    // We'll focus on Browse level first (gate + embed + down_meta) to fix the E4B OOM.

    // Final checksums
    let config_text = std::fs::read_to_string(output_dir.join("index.json"))?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    config.checksums = crate::format::checksums::compute_checksums(output_dir).ok();
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    Ok(())
}

/// Get a tensor from mmap'd GGUF, dequantizing to f32.
fn get_tensor_f32(
    mmap: &memmap2::Mmap,
    data_offset: u64,
    info: &larql_models::loading::gguf::GgufTensorInfo,
) -> Result<Array2<f32>, VindexError> {
    let abs_offset = data_offset + info.offset;
    let n_elements: u64 = info.dims.iter().product();

    let data_size = larql_models::quant::ggml::tensor_data_size(info.tensor_type, n_elements as usize)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    
    let raw = &mmap[abs_offset as usize..abs_offset as usize + data_size];
    let floats = dequantize(raw, info.tensor_type, n_elements as usize)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    match info.n_dims {
        2 => {
            let ne0 = info.dims[0] as usize; // cols
            let ne1 = info.dims[1] as usize; // rows
            Array2::from_shape_vec((ne1, ne0), floats)
                .map_err(|e| VindexError::Parse(format!("tensor {}: {}", info.name, e)))
        }
        1 => {
            let n = info.dims[0] as usize;
            Array2::from_shape_vec((n, 1), floats)
                .map_err(|e| VindexError::Parse(format!("tensor {}: {}", info.name, e)))
        }
        _ => Err(VindexError::Parse(format!("unsupported tensor dims: {}", info.n_dims))),
    }
}

fn normalize_key(name: &str, prefixes: &[&str]) -> String {
    let normalized = normalize_gguf_key(name);
    for prefix in prefixes {
        if let Some(stripped) = normalized.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    normalized
}

fn write_floats(w: &mut impl Write, data: &[f32], dtype: StorageDtype) -> Result<u64, VindexError> {
    let bytes = crate::config::dtype::encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}
