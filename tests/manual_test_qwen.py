import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from llm_hallucination.qwen_instrumented import agent1_trace
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_trace():
    print("Starting Qwen Trace Test...")
    question = "Why is the sky blue?"
    
    try:
        trace = agent1_trace(question)
        print("\nTrace Generation Successful!")
        print("-" * 50)
        print(f"Question: {trace['question']}")
        print(f"Answer: {trace['generator']['answer'][:100]}...")
        
        acts = trace['generator']['model_activations']
        print(f"Layer Norms (count): {len(acts['layer_norms'])}")
        print(f"Attention Entropy (count): {len(acts['attention_entropy'])}")
        print(f"Prompt vs Answer Attn: {acts['prompt_vs_answer_attention']}")
        
        analysis = trace['analysis']
        print(f"CoT Samples: {len(analysis['cot_samples'])}")
        print(f"Prompt Sensitivity: {analysis['prompt_sensitivity']}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during trace generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trace()
