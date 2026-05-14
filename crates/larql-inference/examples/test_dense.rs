fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    println!("Loading model...");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    let tokenizer = larql_inference::load_tokenizer(model_path).unwrap();
    
    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids();
    
    println!("Running dense inference...");
    let result = larql_inference::predict(&weights, &tokenizer, token_ids, 10);
    
    println!("Predictions:");
    for entry in &result.predictions {
        println!("  {} (logit: {:.2})", entry.0, entry.1);
    }
}
