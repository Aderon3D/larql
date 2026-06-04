use std::path::Path;

fn run_test(weights: &larql_inference::ModelWeights, tokenizer: &larql_inference::tokenizers::Tokenizer, prompt: &str) {
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids();
    println!("\nPrompt: '{}'", prompt);
    let result = larql_inference::predict(weights, tokenizer, token_ids, 5);
    println!("Predictions:");
    for entry in &result.predictions {
        println!("  {} (logit: {:.2})", entry.0, entry.1);
    }
}

fn main() {
    let model_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-preliterate.gguf";
    println!("Loading preliterate model...");
    let weights = larql_inference::load_model_dir(Path::new(model_path)).unwrap();
    let tokenizer = larql_inference::load_tokenizer(Path::new(model_path)).unwrap();
    
    // Test Target: Counting Sequence
    run_test(&weights, &tokenizer, "One, two, three,");
    
    // Test Control: Factual Recall
    run_test(&weights, &tokenizer, "The capital of France is");
}
