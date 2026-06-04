use std::path::Path;
use larql_models::loading::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    let gguf = GgufFile::open(path)?;
    
    println!("--- Metadata ---");
    for (k, v) in &gguf.metadata {
        if k.contains("alignment") || k.contains("vocab") || k.contains("token") || k.contains("embedding") {
            println!("{}: {:?}", k, v);
        }
    }
    
    for info in &gguf.tensor_infos {
        if info.name == "token_embd.weight" {
            println!("\nTENSOR: {}: dims={:?}, type={}, offset={}", info.name, info.dims, info.tensor_type, info.offset);
        }
    }
    
    Ok(())
}
