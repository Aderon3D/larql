use larql_inference::TracePositions;
use larql_inference::trace_residuals;
use larql_models::loading::gguf::load_gguf;
use std::path::Path;

fn main() {
    let model_path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf");
    println!("Loading GGUF model...");
    let weights = load_gguf(model_path).expect("Failed to load GGUF model weights");

    println!("Loading tokenizer...");
    let tok_path = Path::new("C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF");
    let tokenizer = larql_inference::load_tokenizer(tok_path).expect("failed tokenizer");

    let prompt = "The letter after B is";
    let answer = " C";
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let token_ids = encoding.get_ids().to_vec();

    let ans_enc = tokenizer.encode(format!(" {}", answer), true).unwrap();
    let answer_id = *ans_enc.get_ids().last().unwrap();

    let ffn = larql_inference::WeightFfn { weights: &weights };
    
    println!("Tracing...");
    let trace = trace_residuals(&weights, &token_ids, TracePositions::Last, false, &ffn);

    let traj = trace.answer_trajectory(&weights, answer_id);
    println!("Answer trajectory for '{}':", answer);
    println!("  {:>5} {:>6} {:>8} {:>9} {:>9} {:>8}", "Layer", "Rank", "Prob", "Attn", "FFN", "Who");

    for w in &traj {
        let who = if w.layer == -1 {
            "embed"
        } else if w.attn_logit > 0.5 && w.ffn_logit > 0.5 {
            "BOTH ↑"
        } else if w.attn_logit > 0.5 {
            "ATTN ↑"
        } else if w.ffn_logit > 0.5 {
            "FFN ↑"
        } else if w.attn_logit < -0.5 && w.ffn_logit < -0.5 {
            "both ↓"
        } else if w.attn_logit < -0.5 {
            "attn ↓"
        } else if w.ffn_logit < -0.5 {
            "ffn ↓"
        } else {
            "~"
        };
        let layer_str = if w.layer == -1 { "emb".to_string() } else { format!("L{}", w.layer) };
        println!("  {:>5} {:>6} {:>8.4} {:>+9.1} {:>+9.1} {:>8}", layer_str, w.rank, w.prob, w.attn_logit, w.ffn_logit, who);
    }
}
