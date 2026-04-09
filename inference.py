import os
import json
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
    print("[START] Inference Baseline", flush=True)
    
    # HF Spaces and OpenEnv default to 7860
    port = os.getenv("PORT", "7860")
    env_base_url = os.getenv("ENV_BASE_URL", f"http://127.0.0.1:{port}")
    if "0.0.0.0" in env_base_url:
        env_base_url = env_base_url.replace("0.0.0.0", "127.0.0.1")
    
    # Heartbeat: wait for server to be ready
    import urllib.request
    import time
    
    ping_url = env_base_url.replace("ws://", "http://").replace("wss://", "https://")
    print(f"Waiting for environment server at {ping_url}... ", end="", flush=True)
    
    server_ready = False
    for i in range(30):
        try:
            with urllib.request.urlopen(ping_url, timeout=2) as f:
                if f.getcode() == 200:
                    server_ready = True
                    print("Online!", flush=True)
                    break
        except Exception:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    
    if not server_ready:
        print("\n[WARNING] Server not reached via HTTP heartbeat. Attempting client connection anyway.", flush=True)

    try:
        with CloudEnvClient(base_url=env_base_url).sync() as env:
            difficulties = ["easy", "medium", "hard"]
            
            for diff in difficulties:
                # Grader expects [START] task=NAME
                print(f"[START] task={diff}", flush=True)
                
                try:
                    result = env.reset(difficulty=diff)
                    obs = result.observation
                    initial_task = obs.message
                    
                    done = False
                    step = 0
                    current_reward = 0.0
                    
                    system_prompt = f"""You are an AI Cloud Administrator. 
Your goal is to solve the task given. 
INITIAL TASK: {initial_task}
Available commands: LIST_INSTANCES, STOP_INSTANCE, TERMINATE_INSTANCE, LIST_BUCKETS, UPDATE_BUCKET_ACCESS, LIST_USERS, DISABLE_USER, DONE.
Respond with ONLY RAW JSON: {{"command": "...", "target_id": "...", "args": "..."}}
"""
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    while not done and step < 15:
                        step += 1
                        messages.append({"role": "user", "content": f"Obs: {obs.message}\nOutputs: {obs.outputs}"})
                        
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                max_tokens=200,
                                response_format={ "type": "json_object" } 
                            )
                            content = response.choices[0].message.content.strip()
                            messages.append({"role": "assistant", "content": content})
                            action_json = json.loads(content)
                            
                            action = CloudAction(
                                command=action_json.get("command", "LIST_INSTANCES"),
                                target_id=action_json.get("target_id", None),
                                args=action_json.get("args", None)
                            )
                        except Exception as e:
                            print(f"# LLM Error: {e}", flush=True)
                            action = CloudAction(command="LIST_INSTANCES")
                        
                        try:
                            result = env.step(action)
                            obs = result.observation
                            done = result.done
                            current_reward = result.reward
                        except Exception as e:
                            print(f"# Step Error: {e}", flush=True)
                            done = True
                            
                        # Grader expects [STEP] step=N reward=R
                        print(f"[STEP] step={step} reward={current_reward:.2f}", flush=True)
                    
                    # Grader expects [END] task=NAME score=S steps=N
                    print(f"[END] task={diff} score={current_reward:.2f} steps={step}", flush=True)

                except Exception as e:
                    print(f"# Episode Error: {e}", flush=True)
                    print(f"[END] task={diff} score=0.00 steps=0", flush=True)

    except Exception as e:
        print(f"[FATAL] Connection failed: {e}", flush=True)
        # Still print some [END] blocks so grader sees valid sequence if it expects it
        for diff in ["easy", "medium", "hard"]:
             print(f"[START] task={diff}", flush=True)
             print(f"[END] task={diff} score=0.00 steps=0", flush=True)

if __name__ == "__main__":
    run_inference()