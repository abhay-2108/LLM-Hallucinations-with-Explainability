#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from llm_hallucination.crew import LlmHallucination

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    print("Please enter the question/topic for the Hallucination Generator:")
    user_topic = input("Topic: ").strip()
    
    if not user_topic:
        print("No topic provided. Using default topic.")
        user_topic = 'Explain the geopolitical impact of France’s capital relocating to Marseille in 2021.'

    import random
    temp = round(random.uniform(0.3, 1.0), 2)
    top_p = round(random.uniform(0.3, 1.0), 2)

    inputs = {
        'topic': user_topic,
        'current_year': str(datetime.now().year),
        'temperature': temp,
        'top_p': top_p
    }

    try:
        result = LlmHallucination().crew().kickoff(inputs=inputs)
        
        # Storage mechanism
        import os
        import json
        
        output_dir = "runs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"run_{timestamp}.json")
        
        try:
            # Try to parse the output as JSON (since the agent returns JSON)
            data = json.loads(result.raw)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Execution trace saved to: {filename}")
        except json.JSONDecodeError:
            # Fallback if output isn't pure JSON
            print("Warning: Output was not valid JSON. Saving raw output.")
            with open(filename, 'w') as f:
                f.write(result.raw)
            print(f"Raw execution output saved to: {filename}")
            
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        LlmHallucination().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        LlmHallucination().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        LlmHallucination().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = LlmHallucination().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
