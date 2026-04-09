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
    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
    if "0.0.0.0" in env_base_url:
        env_base_url = env_base_url.replace("0.0.0.0", "127.0.0.1")
    
    with CloudEnvClient(base_url=env_base_url).sync() as env:
        difficulties = ["easy", "medium", "hard"]
        
        for diff in difficulties:
            # STRICT REGEX LOG: START
            print(f"[START] task={diff}", flush=True)
            try:
                result = env.reset(difficulty=diff)
                obs = result.observation
                initial_task = obs.message
            except Exception as e:
                print(f"[END] task={diff} score=0.01 steps=0", flush=True)
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
                step += 1
                messages.append({
                    "role": "user", 
                    "content": f"Message: {obs.message}\nPrevious Command Output: {obs.outputs}"
                })
                
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
                except Exception:
                    action = CloudAction(command="LIST_INSTANCES")
                
                try:
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    current_reward = result.reward
                except Exception:
                    done = True
                    current_reward = 0.01
                    
                # STRICT REGEX LOG: STEP
                print(f"[STEP] step={step} reward={current_reward:.2f}", flush=True)
                
            # STRICT REGEX LOG: END
            print(f"[END] task={diff} score={current_reward:.2f} steps={step}", flush=True)

if __name__ == "__main__":
    try:
        run_inference()
    except Exception:
        pass