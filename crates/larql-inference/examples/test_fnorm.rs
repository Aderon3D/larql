fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    
    if let Some(w) = weights.vectors.get("norm.weight") {
        println!("Final norm weights (first 10):");
        for i in 0..10 {
            println!("  [{}] {}", i, w[i]);
        }
        let sum: f32 = w.iter().sum();
        let avg = sum / w.len() as f32;
        println!("  Average: {}", avg);
    } else {
        println!("Final norm weight NOT FOUND.");
    }
}
