fn run_test(weights: &larql_inference::ModelWeights, tokenizer: &larql_inference::tokenizers::Tokenizer, prompt: &str) {
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids().to_vec();
    println!("\nPrompt: '{}'", prompt);
    let result = larql_inference::predict(weights, tokenizer, &token_ids, 5);
    println!("Predictions:");
    for (token, prob) in &result.predictions {
        println!("  {} ({:.2}%)", token, prob * 100.0);
    }
}

fn main() {
    let model_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf";
    println!("Loading ablated model directly...");
    let weights = larql_models::loading::gguf::load_gguf(std::path::Path::new(model_path)).unwrap();
    println!("{:#?}", weights.arch.config());
    let tokenizer = larql_inference::load_tokenizer(std::path::Path::new(model_path)).unwrap();
    
    // Math probes
    run_test(&weights, &tokenizer, "The act of reciting numbers in order is called");
    run_test(&weights, &tokenizer, "Addition, subtraction, and multiplication are forms of");
    run_test(&weights, &tokenizer, "If you have five apples and eat two, how many remain");
    
    // Control probes
    run_test(&weights, &tokenizer, "The capital of France is");
    run_test(&weights, &tokenizer, "A haiku is a form of poetry that");
}
