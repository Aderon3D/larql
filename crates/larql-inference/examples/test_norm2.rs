fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    
    let pre_ffn = "layers.12.pre_feedforward_layernorm.weight";
    if let Some(w) = weights.vectors.get(pre_ffn) {
        let sum: f32 = w.iter().sum();
        let avg = sum / w.len() as f32;
        println!("Pre-FFN norm layer 12 weights avg: {}", avg);
    }
    
    let post_attn = "layers.12.post_attention_layernorm.weight";
    if let Some(w) = weights.vectors.get(post_attn) {
        let sum: f32 = w.iter().sum();
        let avg = sum / w.len() as f32;
        println!("Post-Attn norm layer 12 weights avg: {}", avg);
    }
}
