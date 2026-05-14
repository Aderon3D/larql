fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let weights = larql_inference::load_model_dir(model_path).unwrap();
    let tokenizer = larql_inference::load_tokenizer(model_path).unwrap();
    
    let prompt = "<start_of_turn>user\nThe capital of France is<end_of_turn>\n<start_of_turn>model\n";
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids();
    
    println!("Running dense inference...");
    let result = larql_inference::predict(&weights, &tokenizer, token_ids, 20);
    
    println!("Predictions:");
    for (i, (token, logit)) in result.predictions.iter().enumerate() {
        println!("  {}: {} (logit: {:.2})", i + 1, token, logit);
    }
}
