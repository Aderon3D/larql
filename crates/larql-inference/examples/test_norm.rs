fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    
    let norm_k = "layers.0.input_layernorm.weight";
    if let Some(w) = weights.vectors.get(norm_k) {
        println!("Input layernorm layer 0 weights:");
        for i in 0..10 {
            println!("  [{}] {}", i, w[i]);
        }
        let sum: f32 = w.iter().sum();
        let avg = sum / w.len() as f32;
        println!("  Average: {}", avg);
    }
    
    let qk_norm = "layers.0.self_attn.q_norm.weight";
    if let Some(w) = weights.vectors.get(qk_norm) {
        println!("QK norm layer 0 weights:");
        for i in 0..10 {
            println!("  [{}] {}", i, w[i]);
        }
        let sum: f32 = w.iter().sum();
        let avg = sum / w.len() as f32;
        println!("  Average: {}", avg);
    }
}
