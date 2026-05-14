fn main() {
    let model_path = std::path::Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    let file = std::fs::File::open(model_path).unwrap();
    let mut reader = std::io::BufReader::new(file);
    let container = gguf_rs::GGUFContainer::read(&mut reader).unwrap();
    
    println!("GGUF Metadata Keys:");
    for (key, _) in container.metadata() {
        println!("  {}", key);
    }
}
