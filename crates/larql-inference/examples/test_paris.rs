use std::path::Path;
use larql_inference::{
    load_model_dir, load_tokenizer, predict_with_graph, WalkLayerGraph,
    WalkFfn, cpu_backend,
};
use larql_vindex::{VectorIndex, SilentLoadCallbacks};

fn main() {
    let model_path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf");
    let vindex_path = Path::new("C:/Users/alby9/Documents/Antygravity/Ai World/larql/ablation_project/indices/gemma-4-E4B-surgical.vindex");
    
    println!("Loading weights (lazy embedding enabled)...");
    let weights = load_model_dir(model_path).expect("Failed to load weights");
    
    println!("Loading tokenizer...");
    let tokenizer_path = Path::new("../../gemma-tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .or_else(|_| tokenizers::Tokenizer::from_file("gemma-tokenizer.json"))
        .expect("Failed to load tokenizer");
    
    println!("Loading VIndex from {:?}...", vindex_path);
    let mut callbacks = SilentLoadCallbacks;
    let vindex = VectorIndex::load_vindex(vindex_path, &mut callbacks).expect("Failed to load VIndex");
    
    let backend = cpu_backend();
    let walk_ffn = WalkFfn::new_unlimited_with_backend(&weights, &vindex, &*backend);
    let graph = WalkLayerGraph { ffn: &walk_ffn, backend: Some(&*backend) };
    println!("Model Metadata:");
    println!("  Layers: {}", weights.num_layers);
    println!("  Hidden: {}", weights.hidden_size);
    println!("  Vocab:  {}", weights.vocab_size);
    println!("  Head Dim: {}", weights.head_dim);
    println!("  LM Head Shape: {:?}", weights.lm_head.shape());
    
    let graph = WalkLayerGraph {
        ffn: &walk_ffn,
        backend: Some(&*backend),
    };

    let prompt = "The capital of France is P";
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids();
    println!("Prompt Tokens: {:?}", token_ids);
    
    let result = predict_with_graph(&weights, &tokenizer, token_ids, 5, &graph);
    
    println!("Predictions:");
    for (i, (s, p)) in result.predictions.iter().enumerate() {
        println!("  {}: {} (prob: {:.4})", i + 1, s, p);
    }
}
