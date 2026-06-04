use std::path::Path;
use larql_models::loading::gguf::load_gguf_selective;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    
    println!("Loading model (selective, skipping FFN) to verify embedding stabilization...");
    let weights = load_gguf_selective(path, true)?;
    
    if let Some(lazy) = &weights.lazy_embed {
        println!("Lazy embedding detected. Type: {}, Vocab: {}, Hidden: {}", 
                 lazy.tensor_type, lazy.vocab_size, lazy.hidden_size);
        
        let token_id = 2; // The problematic token
        println!("Dequantizing token ID {}...", token_id);
        
        let floats = lazy.dequantize_row(token_id)?;
        
        let nan_count = floats.iter().filter(|&&f| !f.is_finite()).count();
        if nan_count > 0 {
            println!("FAILURE: Found {} non-finite values in token {} embedding!", nan_count, token_id);
            for (i, &f) in floats.iter().enumerate().take(10) {
                println!("  [{}] = {}", i, f);
            }
        } else {
            println!("SUCCESS: Token {} dequantized perfectly with no NaNs/Infs.", token_id);
            println!("Sample values: {:?}", &floats[..8]);
        }
    } else {
        println!("ERROR: Lazy embedding table not found in weights.");
    }
    
    Ok(())
}
