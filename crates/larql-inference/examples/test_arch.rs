fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    let arch = &*weights.arch;
    
    println!("Layer Types:");
    for i in 0..42 {
        let hd = arch.head_dim_for_layer(i);
        let rb = arch.rope_base_for_layer(i);
        println!("  Layer {}: head_dim={}, rope_base={}", i, hd, rb);
    }
}
