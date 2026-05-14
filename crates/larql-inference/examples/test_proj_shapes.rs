fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    let arch = &*weights.arch;
    
    for layer in [0, 5].iter() {
        let q_key = arch.attn_q_key(*layer);
        let k_key = arch.attn_k_key(*layer);
        let v_key = arch.attn_v_key(*layer);
        let o_key = arch.attn_o_key(*layer);
        
        let q_shape = weights.tensors.get(&q_key).unwrap().shape();
        let k_shape = weights.tensors.get(&k_key).unwrap().shape();
        let v_shape = weights.tensors.get(&v_key).unwrap().shape();
        let o_shape = weights.tensors.get(&o_key).unwrap().shape();
        
        println!("Layer {}:", layer);
        println!("  Q: {:?}", q_shape);
        println!("  K: {:?}", k_shape);
        println!("  V: {:?}", v_shape);
        println!("  O: {:?}", o_shape);
    }
}
