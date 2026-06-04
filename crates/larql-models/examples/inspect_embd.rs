use std::path::Path;
use larql_models::loading::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    let gguf = GgufFile::open(path)?;
    
    for info in &gguf.tensor_infos {
        if info.name == "token_embd.weight" {
            println!("TENSOR: {}: dims={:?}, type={}", info.name, info.dims, info.tensor_type);
        }
    }
    
    Ok(())
}
