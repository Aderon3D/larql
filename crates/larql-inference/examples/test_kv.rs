fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    let arch = &*weights.arch;
    
    println!("Checking KV weights for layer 23 (source) vs layer 24 (shared):");
    
    for layer in 23..25 {
        let k_key = arch.attn_k_key(layer);
        if let Some(w) = weights.tensors.get(&k_key) {
            let sum: f32 = w.iter().sum();
            println!("Layer {} K weights exist. Sum: {}", layer, sum);
        } else {
            println!("Layer {} K weights DO NOT EXIST.", layer);
        }
    }
}
