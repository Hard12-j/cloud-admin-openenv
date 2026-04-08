import os
import json
import time
from openai import OpenAI
from client import CloudEnvClient
from models import CloudAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy"
)

def run_inference():
    print("[START] Inference Baseline")
    
    # We use a mocked/local server URL if not provided by hackathon grader
    # The hackathon usually passes the URL or starts it nearby
    port = os.getenv("PORT", "7860")
    env_base_url = os.getenv("ENV_BASE_URL", f"http://127.0.0.1:{port}")
    if "0.0.0.0" in env_base_url:
        env_base_url = env_base_url.replace("0.0.0.0", "127.0.0.1")
    
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            with CloudEnvClient(base_url=env_base_url).sync() as env:
                # We will loop through the difficulties as tests
                difficulties = ["easy", "medium", "hard"]
                
                for diff in difficulties:
                    print(f"[START] Episode {diff}")
                    try:
                        result = env.reset(difficulty=diff)
                        obs = result.observation
                        initial_task = obs.message
                    except Exception as e:
                        print(f"Error resetting environment: {e}")
                        continue
                        
                    done = False
                    step = 0
                    
                    system_prompt = f"""You are an AI Cloud Administrator. 
Your goal is to solve the task given. 

INITIAL TASK: {initial_task}

You can use the following commands:
- LIST_INSTANCES
- STOP_INSTANCE (requires target_id)
- TERMINATE_INSTANCE (requires target_id)
- LIST_BUCKETS
- UPDATE_BUCKET_ACCESS (requires target_id and args='private' or 'public')
- LIST_USERS
- DISABLE_USER (requires target_id)
- DONE

You must respond with ONLY a valid JSON object holding your action. 
Valid JSON schema:
{{
  "command": "COMMAND_NAME",
  "target_id": "optional_target_id",
  "args": "optional_args"
}}
DO NOT wrapping your response in Markdown codeblocks. Return RAW JSON.
Call the DONE command once you verify the task is fully accomplished!
"""
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    while not done and step < 15:
                        print(f"[STEP] {step} Obs Message: {obs.message}")
                        
                        messages.append({
                            "role": "user", 
                            "content": f"Message: {obs.message}\nPrevious Command Output: {obs.outputs}"
                        })
                        
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                max_tokens=200,
                                response_format={ "type": "json_object" } # Ensure JSON mode if supported
                            )
                            content = response.choices[0].message.content.strip()
                            messages.append({"role": "assistant", "content": content})
                            action_json = json.loads(content)
                            
                            action = CloudAction(
                                command=action_json.get("command", "LIST_INSTANCES"),
                                target_id=action_json.get("target_id", None),
                                args=action_json.get("args", None)
                            )
                            print(f"Agent chose action: {action.command} target_id={action.target_id} args={action.args}")
                        except Exception as e:
                            print(f"LLM Error: {e}, defaulting to LIST_INSTANCES")
                            action = CloudAction(command="LIST_INSTANCES")
                        
                        try:
                            result = env.step(action)
                            obs = result.observation
                            done = result.done
                        except Exception as e:
                            print(f"Error stepping environment: {e}")
                            break
                        step += 1
                        
                    if 'result' in locals() and hasattr(result, 'reward'):
                        print(f"[END] Episode {diff} - Score: {result.reward}")
                    else:
                        print(f"[END] Episode {diff} failed.")
            
            # If we succeed the full loop without crashing the client connection, break retry loop
            break
            
        except Exception as e:
            print(f"ConnectionError on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting inference script.")
                # We exit cleanly with return code 0 but we've logged the failure
                # or we can let it gracefully finish. The exception log will tell the grader.

if __name__ == "__main__":
    run_inference()
