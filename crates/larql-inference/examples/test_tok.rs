fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let tokenizer = larql_inference::load_tokenizer(model_path).unwrap();
    
    let prompt = "<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n";
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids();
    
    println!("Tokens:");
    for &id in token_ids {
        let dec = tokenizer.decode(&[id], true).unwrap();
        let raw_dec = larql_inference::decode_token_raw(&tokenizer, id);
        println!("  {}: {} (raw: {})", id, dec, raw_dec);
    }
}
