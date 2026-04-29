use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use serde::Deserialize;

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
    println!("Starting BULK GGUF ablation (L15-L35)...");
    
    let base_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf";
    let output_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-BLANK.gguf";

    // Open GGUF
    let gguf = larql_models::loading::gguf::GgufFile::open(std::path::Path::new(base_path))?;

    // Copy file
    if !std::path::Path::new(output_path).exists() {
        println!("Copying base model...");
        std::fs::copy(base_path, output_path)?;
    }

    let mut out_file = OpenOptions::new().write(true).open(output_path)?;
    let base_offset = gguf.data_offset;

    let mut ablated_count = 0;
    for layer in 15..=35 {
        let gate_key = format!("blk.{}.ffn_gate.weight", layer);
        let up_key = format!("blk.{}.ffn_up.weight", layer);

        for key in [gate_key, up_key] {
            if let Some(tensor) = gguf.tensor_infos.iter().find(|t| t.name == key) {
                let cols = tensor.dims[0] as usize;
                let rows = if tensor.dims.len() > 1 { tensor.dims[1] as usize } else { 1 };
                
                let bytes_per_row = tensor_data_size(tensor.tensor_type, cols);
                let total_bytes = rows as u64 * bytes_per_row as u64;
                
                let absolute_offset = base_offset + tensor.offset;
                out_file.seek(SeekFrom::Start(absolute_offset))?;
                
                // Zero out the entire tensor data
                let chunk_size = 1024 * 1024;
                let zeros = vec![0u8; chunk_size];
                let mut remaining = total_bytes;
                while remaining > 0 {
                    let to_write = remaining.min(chunk_size as u64);
                    out_file.write_all(&zeros[..to_write as usize])?;
                    remaining -= to_write;
                }
                println!("Zeroed out tensor {} ({} bytes)", key, total_bytes);
                ablated_count += 1;
            }
        }
    }
    
    println!("Done. Zeroed out {} tensors.", ablated_count);
    Ok(())
}
