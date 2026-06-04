from llama_cpp import Llama

def test_ablated_model():
    model_path = "C:/Users/alby9/.cache/lm-studio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf"
    
    print("Loading model via llama.cpp...")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_gpu_layers=0,  # CPU only to avoid any VRAM issues
        verbose=False
    )
    
    prompts = [
        "If you have five apples and eat two, how many remain?",
        "Count from 1 to 5: 1, 2, 3,",
        "The capital of France is",
        "A poem about a tree:"
    ]
    
    for prompt in prompts:
        print(f"\n--- PROMPT ---\n{prompt}")
        output = llm(
            prompt,
            max_tokens=20,
            temperature=0.0,
            echo=False
        )
        print(f"RESPONSE: {output['choices'][0]['text'].strip()}")

if __name__ == "__main__":
    test_ablated_model()
