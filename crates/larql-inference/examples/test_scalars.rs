fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    
    for layer in 0..42 {
        let key = format!("layers.{}.layer_scalar", layer);
        if let Some(v) = weights.vectors.get(&key) {
            println!("Layer {} scalar: {}", layer, v[0]);
        }
    }
}
