//! LarQL Split Format loader — reads model weights from a directory with weight_manifest.json.
//! This format is produced by `COMPILE INTO MODEL` and used for edited models.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::weights::ModelWeights;
use crate::detect::ModelError;

#[derive(Serialize, Deserialize)]
struct WeightEntry {
    key: String,
    kind: String,
    shape: Vec<usize>,
    offset: u64,
    length: u64,
    #[serde(default)]
    file: String,
}

/// Load model weights from a directory containing weight_manifest.json.
pub fn load_split_model(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    let path = path.as_ref();
    let manifest_path = path.join("weight_manifest.json");
    let index_path = path.join("index.json");

    if !manifest_path.exists() {
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    // Load manifest
    let manifest_text = std::fs::read_to_string(&manifest_path)?;
    let entries: Vec<WeightEntry> = serde_json::from_str(&manifest_text)
        .map_err(|e| ModelError::Parse(format!("weight_manifest.json: {e}")))?;

    // Detect architecture from index.json (vindex-style config)
    if !index_path.exists() {
        return Err(ModelError::Parse("index.json not found in split model directory".into()));
    }
    let index_text = std::fs::read_to_string(&index_path)?;
    let index_json: serde_json::Value = serde_json::from_str(&index_text)
        .map_err(|e| ModelError::Parse(format!("index.json: {e}")))?;

    let _hidden_size = index_json["hidden_size"].as_u64().unwrap_or(0) as usize;
    let _num_layers = index_json["num_layers"].as_u64().unwrap_or(0) as usize;
    let _intermediate_size = index_json["intermediate_size"].as_u64().unwrap_or(0) as usize;
    let vocab_size = index_json["vocab_size"].as_u64().unwrap_or(0) as usize;

    // Detect architecture from the same JSON
    let arch = crate::detect_from_json(&index_json);

    let mut mmap_cache: HashMap<String, memmap2::Mmap> = HashMap::new();
    let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_loaded: Option<crate::WeightArray> = None;
    let mut embed_loaded: Option<crate::WeightArray> = None;

    for entry in &entries {
        let filename = if entry.file.is_empty() { "model_weights.bin".to_string() } else { entry.file.clone() };

        if !mmap_cache.contains_key(&filename) {
            let fpath = path.join(&filename);
            if fpath.exists() {
                let f = std::fs::File::open(&fpath)?;
                let m = unsafe { memmap2::Mmap::map(&f)? };
                mmap_cache.insert(filename.clone(), m);
            }
        }

        let data = match mmap_cache.get(&filename) {
            Some(m) => m.as_ref(),
            None => continue,
        };

        let byte_offset = entry.offset as usize;
        let byte_count = entry.length as usize;
        if byte_offset + byte_count > data.len() {
            return Err(ModelError::Parse(format!("Weight entry {}: offset out of bounds", entry.key)));
        }

        let raw_bytes = &data[byte_offset..byte_offset + byte_count];
        let expected_floats: usize = entry.shape.iter().product();
        
        // Split format always encodes as the storage type specified in the build (usually f32 in compiled models)
        // or matches byte count / floats.
        let floats = if byte_count == expected_floats * 4 {
            raw_bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        } else if byte_count == expected_floats * 2 {
            crate::quant::half::decode_f16(raw_bytes)
        } else {
            return Err(ModelError::Parse(format!("Weight entry {}: size mismatch", entry.key)));
        };

        match entry.kind.as_str() {
            "tensor" => {
                let arr = Array2::from_shape_vec((entry.shape[0], entry.shape[1]), floats)
                    .map_err(|e| ModelError::Parse(e.to_string()))?;
                let shared = arr.into_shared();
                if entry.key == "lm_head.weight" {
                    lm_head_loaded = Some(shared);
                } else if entry.key == arch.embed_key() {
                    embed_loaded = Some(shared.clone());
                    tensors.insert(entry.key.clone(), shared);
                } else {
                    tensors.insert(entry.key.clone(), shared);
                }
            }
            "vector" => {
                vectors.insert(entry.key.clone(), floats);
            }
            _ => {}
        }
    }

    let embed = embed_loaded.ok_or_else(|| ModelError::MissingTensor(arch.embed_key().into()))?;
    let lm_head = lm_head_loaded.unwrap_or_else(|| embed.clone());

    let cfg = arch.config();
    Ok(ModelWeights {
        tensors, vectors, embed, lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}
