import os
import json
import time
import urllib.request
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

def strict_score(value) -> float:
    """
    Guarantee the returned score is always strictly between 0 and 1.
    Never allow 0.0 or 1.0.
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        x = 0.01

    if x <= 0.0:
        x = 0.01
    elif x >= 1.0:
        x = 0.99

    return min(0.99, max(0.01, x))

def run_inference():
    # OpenEnv / HF Space usually runs on 7860
    port = os.getenv("PORT", "7860")
    env_base_url = os.getenv("ENV_BASE_URL", f"http://127.0.0.1:{port}")

    if "0.0.0.0" in env_base_url:
        env_base_url = env_base_url.replace("0.0.0.0", "127.0.0.1")

    ping_url = env_base_url.replace("ws://", "http://").replace("wss://", "https://")

    # Wait briefly for server startup
    server_ready = False
    for _ in range(30):
        try:
            with urllib.request.urlopen(ping_url, timeout=2) as response:
                if response.getcode() == 200:
                    server_ready = True
                    break
        except Exception:
            pass
        time.sleep(2)

    try:
        with CloudEnvClient(base_url=env_base_url).sync() as env:
            for diff in ["easy", "medium", "hard"]:
                print(f"[START] task={diff}", flush=True)

                step = 0
                current_reward = 0.01

                try:
                    result = env.reset(difficulty=diff)
                    obs = result.observation
                    done = False

                    system_prompt = f"""You are an AI Cloud Administrator.
Solve the cloud administration task safely and efficiently.

TASK:
{obs.message}

Available commands:
- LIST_INSTANCES
- STOP_INSTANCE
- TERMINATE_INSTANCE
- LIST_BUCKETS
- UPDATE_BUCKET_ACCESS
- LIST_USERS
- DISABLE_USER
- DONE

Return ONLY valid JSON in this format:
{{"command": "COMMAND_NAME", "target_id": "optional_id_or_null", "args": "optional_args_or_null"}}
"""

                    messages = [{"role": "system", "content": system_prompt}]

                    while not done and step < 15:
                        step += 1

                        messages.append({
                            "role": "user",
                            "content": f"Observation: {obs.message}\nOutputs: {obs.outputs}"
                        })

                        action = CloudAction(command="DONE", target_id=None, args=None)

                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=messages,
                                max_tokens=200,
                                response_format={"type": "json_object"}
                            )

                            content = (response.choices[0].message.content or "").strip()
                            messages.append({"role": "assistant", "content": content})

                            action_json = json.loads(content)

                            action = CloudAction(
                                command=action_json.get("command", "DONE"),
                                target_id=action_json.get("target_id"),
                                args=action_json.get("args")
                            )
                        except Exception:
                            # Safe fallback
                            action = CloudAction(command="DONE", target_id=None, args=None)

                        try:
                            result = env.step(action)
                            obs = result.observation
                            done = bool(result.done)
                            current_reward = strict_score(result.reward)
                        except Exception:
                            done = True
                            current_reward = 0.01

                        print(f"[STEP] step={step} reward={strict_score(current_reward):.4f}", flush=True)

                    print(f"[END] task={diff} score={strict_score(current_reward):.4f} steps={step}", flush=True)

                except Exception:
                    print(f"[END] task={diff} score=0.0100 steps={step}", flush=True)

    except Exception:
        # Even on total connection failure, still emit valid structured output
        for diff in ["easy", "medium", "hard"]:
            print(f"[START] task={diff}", flush=True)
            print(f"[END] task={diff} score=0.0100 steps=0", flush=True)

if __name__ == "__main__":
    run_inference()