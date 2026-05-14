use std::path::Path;
use larql_models::loading::gguf::load_gguf;

fn main() {
    let path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    println!("Loading GGUF from {:?}...", path);
    match load_gguf(path) {
        Ok(weights) => {
            let mut keys: Vec<String> = weights.tensors.keys().cloned().collect();
            keys.sort();

            println!("\n--- GLOBAL TENSORS ---");
            for k in &keys {
                if !k.contains("layers.") {
                    let t = weights.tensors.get(k).unwrap();
                    println!("  {:30} | shape: {:?}", k, t.shape());
                }
            }

            println!("\n--- LAYER 0 TENSORS ---");
            for k in &keys {
                if k.starts_with("layers.0.") {
                    let t = weights.tensors.get(k).unwrap();
                    println!("  {:30} | shape: {:?}", k, t.shape());
                }
            }

            println!("\n--- VECTOR TENSORS (1D) ---");
            let mut v_keys: Vec<String> = weights.vectors.keys().cloned().collect();
            v_keys.sort();
            for k in v_keys {
                let v = weights.vectors.get(&k).unwrap();
                println!("  {:30} | len: {:?}", k, v.len());
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }
}
