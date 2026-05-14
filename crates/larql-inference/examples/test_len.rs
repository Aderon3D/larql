fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    
    for layer in [0, 5] {
        let q_norm_k = format!("layers.{}.self_attn.q_norm.weight", layer);
        if let Some(w) = weights.vectors.get(&q_norm_k) {
            println!("Layer {} QK norm weight length: {}", layer, w.len());
        } else {
            println!("Layer {} QK norm weight DO NOT EXIST.", layer);
        }
    }
}
