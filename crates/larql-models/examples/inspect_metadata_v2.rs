use std::path::Path;
use larql_models::loading::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    let gguf = GgufFile::open(path)?;
    
    println!("--- Key Metadata ---");
    let keys = [
        "general.alignment",
        "gemma4.block_count",
        "gemma4.embedding_length",
        "gemma4.feed_forward_length",
        "tokenizer.ggml.tokens", // Only check length
    ];
    
    for &k in &keys {
        if let Some(v) = gguf.metadata.get(k) {
            match v {
                larql_models::loading::gguf::GgufValue::Array(a) if k == "tokenizer.ggml.tokens" => {
                    println!("{}: Array(len={})", k, a.len());
                }
                _ => println!("{}: {:?}", k, v),
            }
        }
    }
    
    for info in &gguf.tensor_infos {
        if info.name == "token_embd.weight" {
            println!("\nTENSOR: {}: dims={:?}, type={}, offset={}", info.name, info.dims, info.tensor_type, info.offset);
        }
    }
    
    Ok(())
}
