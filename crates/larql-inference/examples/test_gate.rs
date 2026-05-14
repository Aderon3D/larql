fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    
    let gate_k = "layers.12.mlp.gate_proj.weight";
    if let Some(w) = weights.tensors.get(gate_k) {
        println!("Gate layer 12 weights shape: {:?}", w.shape());
        let slice = w.as_slice().unwrap();
        println!("Gate layer 12 weights (first 20):");
        for i in 0..20 {
            println!("  [{}] {}", i, slice[i]);
        }
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum_sq = 0.0;
        for &v in slice {
            if v < min { min = v; }
            if v > max { max = v; }
            sum_sq += v * v;
        }
        let var = sum_sq / slice.len() as f32;
        println!("  Min: {}, Max: {}, Var: {}", min, max, var);
    } else {
        println!("Gate layer 12 weights DO NOT EXIST.");
    }
}
