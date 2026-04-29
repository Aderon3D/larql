use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct Patch {
    operations: Vec<PatchOp>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
enum PatchOp {
    Insert { layer: usize, feature: usize, #[serde(default)] target: String },
    Update { layer: usize, feature: usize },
    Delete { layer: usize, feature: usize },
    #[serde(other)]
    Other,
}

fn tensor_data_size(tensor_type: u32, n_elements: usize) -> usize {
    match tensor_type {
        0 => n_elements * 4, // F32
        1 => n_elements * 2, // F16
        2 => n_elements / 32 * 18, // Q4_0
        3 => n_elements / 32 * 20, // Q4_1
        8 => n_elements / 32 * 22, // Q5_0
        9 => n_elements / 32 * 24, // Q5_1
        6 => n_elements / 32 * 34, // Q8_0
        12 => n_elements / 256 * 144, // Q4_K
        14 => n_elements / 256 * 210, // Q6_K
        10 => n_elements / 256 * 84, // Q2_K
        11 => n_elements / 256 * 110, // Q3_K
        13 => n_elements / 256 * 176, // Q5_K
        _ => panic!("Unsupported GGML type: {}", tensor_type),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting GGUF ablation...");
    
    let base_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf";
    let patch_path = "C:/Users/alby9/Documents/Antygravity/Ai World/ablation_project/indices/preliterate_semantic_4B.vlp";
    let output_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-preliterate.gguf";

    // 1. Read the patch to find deletions
    let patch_json = std::fs::read_to_string(patch_path).expect("Failed to read patch");
    let patch: Patch = serde_json::from_str(&patch_json).expect("Failed to parse patch");
    
    let mut deletions = std::collections::HashMap::<usize, Vec<usize>>::new();
    let mut total_deletions = 0;
    for op in patch.operations {
        if let PatchOp::Delete { layer, feature } = op {
            deletions.entry(layer).or_default().push(feature);
            total_deletions += 1;
        }
    }
    
    println!("Found {} features to ablate across {} layers.", total_deletions, deletions.len());

    // 2. Open GGUF and get tensors
    let gguf = larql_models::loading::gguf::GgufFile::open(std::path::Path::new(base_path))?;

    // 3. Copy file
    if !std::path::Path::new(output_path).exists() {
        println!("Copying base model to {} ... (this may take a minute)", output_path);
        std::fs::copy(base_path, output_path)?;
    } else {
        println!("Target model already exists, patching in place...");
    }

    // 4. Open output file for patching
    let mut out_file = OpenOptions::new().write(true).open(output_path)?;
    let base_offset = gguf.data_offset;

    // 5. Apply ablations
    let mut ablated_count = 0;
    for (layer, features) in deletions {
        let gate_key = format!("blk.{}.ffn_gate.weight", layer);
        let up_key = format!("blk.{}.ffn_up.weight", layer);

        let mut process_tensor = |key: &str, file: &mut File| -> Result<(), Box<dyn std::error::Error>> {
            if let Some(tensor) = gguf.tensor_infos.iter().find(|t| t.name == key) {
                let cols = tensor.dims[0] as usize;
                let rows = if tensor.dims.len() > 1 { tensor.dims[1] as usize } else { 1 };
                
                // For ffn_gate and ffn_up, row 'i' corresponds to feature 'i'.
                if rows == 10240 && cols == 2560 {
                    let bytes_per_row = tensor_data_size(tensor.tensor_type, cols);
                    for &feature in &features {
                        if feature >= rows {
                            println!("Warning: feature {} out of bounds for tensor {} (rows={})", feature, key, rows);
                            continue;
                        }
                        let absolute_offset = base_offset + tensor.offset + (feature as u64 * bytes_per_row as u64);
                        
                        file.seek(SeekFrom::Start(absolute_offset))?;
                        let zeros = vec![0u8; bytes_per_row];
                        file.write_all(&zeros)?;
                        ablated_count += 1;
                    }
                } else {
                    println!("Warning: tensor {} has unexpected dims [{}, {}]", key, cols, rows);
                }
            } else {
                println!("Warning: tensor {} not found", key);
            }
            Ok(())
        };

        process_tensor(&gate_key, &mut out_file)?;
        process_tensor(&up_key, &mut out_file)?;
    }
    
    println!("Done. Zeroed out {} tensor rows.", ablated_count);
    Ok(())
}
