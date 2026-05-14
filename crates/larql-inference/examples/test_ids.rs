fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let tokenizer = larql_inference::load_tokenizer(model_path).unwrap();
    
    let tokens = ["Paris", " Paris", "极其", "extraordinarily", " extraordinarily", "🍘"];
    for t in tokens {
        let enc = tokenizer.encode(t, false).unwrap();
        println!("{}: {:?}", t, enc.get_ids());
    }
}
