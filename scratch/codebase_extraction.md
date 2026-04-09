# Repository Architecture
```text
cloud_infra_env/
    client.py
    Dockerfile
    inference.py
    models.py
    openenv.yaml
    output.txt
    pyproject.toml
    README.md
    requirements.txt
    scratch/
        codebase_extraction.md
        codebase_extraction_utf8.md
        extract_codebase.py
        verify_clamp.py
    server/
        app.py
        Dockerfile
        environment.py
```

### client.py
```python
from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import CloudAction, CloudObservation, CloudState

class CloudEnvClient(EnvClient[CloudAction, CloudObservation, CloudState]):
    def _step_payload(self, action: CloudAction) -> dict:
        return {
            "command": action.command,
            "target_id": action.target_id,
            "args": action.args
        }

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        done = payload.get("done", False)
        # Ensure reward is never 0.0, even if server returns it
        reward = payload.get("reward")
        if reward is None or (isinstance(reward, (int, float)) and reward <= 0):
            reward = 0.01
        elif isinstance(reward, (int, float)) and reward >= 1.0:
            reward = 0.99
        reward = float(reward)
        
        return StepResult(
            observation=CloudObservation(
                done=done,
                reward=reward,
                message=obs_data.get("message", ""),
                outputs=obs_data.get("outputs", None)
            ),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> CloudState:
        return CloudState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            max_steps=payload.get("max_steps", 15),
            resources=payload.get("resources", {}),
            users=payload.get("users", {})
        )

```

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for Hugging Face Spaces
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose the default HF Spaces port
EXPOSE 7860

# Command to run the server
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}

```

### inference.py
```python
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

def clamp_score(score):
    """Guarantees score is strictly between 0 and 1 by clamping to [0.01, 0.99]"""
    try:
        val = float(score)
        return max(0.01, min(0.99, val))
    except (ValueError, TypeError):
        return 0.01

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
                    current_reward = 0.01
                    
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
                        clamped_r = clamp_score(current_reward)
                        print(f"[STEP] step={step} reward={clamped_r:.2f}", flush=True)
                    
                    # Grader expects [END] task=NAME score=S steps=N
                    clamped_s = clamp_score(current_reward)
                    print(f"[END] task={diff} score={clamped_s:.2f} steps={step}", flush=True)

                except Exception as e:
                    print(f"# Episode Error: {e}", flush=True)
                    score_val = clamp_score(0.01)
                    print(f"[END] task={diff} score={score_val:.2f} steps=0", flush=True)

    except Exception as e:
        print(f"[FATAL] Connection failed: {e}", flush=True)
        for diff in ["easy", "medium", "hard"]:
             print(f"[START] task={diff}", flush=True)
             score_val = clamp_score(0.01)
             print(f"[END] task={diff} score={score_val:.2f} steps=0", flush=True)

if __name__ == "__main__":
    run_inference()
```

### models.py
```python
from typing import List, Optional, Dict, Any
from openenv.core.env_server import Action, Observation, State

class CloudAction(Action):
    command: str  # Action to take, e.g., "LIST_INSTANCES", "STOP_INSTANCE", "DELETE_USER"
    target_id: Optional[str] = None # ID of the target resource, e.g., "i-0123"
    args: Optional[str] = None # Any additional arguments

class CloudObservation(Observation):
    message: str
    outputs: Optional[List[Dict[str, Any]]] = None # Output payload (e.g. list of instances)

class CloudState(State):
    resources: Dict[str, Any] = {}
    users: Dict[str, Any] = {}
    difficulty: str = "easy"
    max_steps: int = 15

```

### openenv.yaml
```yaml
version: 1.0.0
name: "cloud-admin-simulator"
description: "An isolated cloud infrastructure simulation for testing AI admin agents."
server:
  entrypoint: "server.app:app"
  host: "0.0.0.0"
  port: 7860

```

### output.txt
```txt
Error reading file: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "cloud-admin-openenv"
version = "1.0.0"
description = "Cloud Infrastructure Manager OpenEnv simulation"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "openenv-core",
    "fastapi",
    "uvicorn",
    "pydantic",
    "openai"
]

[project.scripts]
server = "server.app:main"

```

### README.md
```markdown
---
title: Cloud Admin OpenEnv
emoji: ☁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Cloud Infrastructure Manager (OpenEnv)

## Description
This is a standard OpenEnv-based project simulating an AWS/GCP administrative task. The agent has tools to list instances, buckets, users, and modify their states in order to resolve configuration and security tasks (like shutting down unwanted instances, disabling compromised users, and changing public bucket accesses).

## Action Space
`CloudAction(command=str, target_id=str, args=str)`
* `command` can be one of: `LIST_INSTANCES`, `STOP_INSTANCE`, `TERMINATE_INSTANCE`, `LIST_BUCKETS`, `UPDATE_BUCKET_ACCESS`, `LIST_USERS`, `DISABLE_USER`, `DONE`.

## Observation Space
`CloudObservation(message=str, outputs=List[dict])`

## Tasks Graded
* **Easy:** Stop a temporary instance based on its tags. (Reward 0.99)
* **Medium:** Ensure an S3 bucket named 'public-assets' has public access disabled. (Reward 0.99)
* **Hard:** Disable a compromised user and terminate their associated instance. (Reward: 0.5 partial, 0.99 full)

## Setup & Run
This environment requires `openenv-core`.

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## AI Inference Baseline
Run the baseline AI agent that interacts with the app locally.

**For Linux/Mac (Bash):**
```bash
export HF_TOKEN="your_groq_api_key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"

python inference.py
```

**For Windows Command Prompt (CMD):**
```cmd
set HF_TOKEN=your_groq_api_key
set API_BASE_URL=https://api.groq.com/openai/v1
set MODEL_NAME=llama-3.1-8b-instant

python inference.py
```

**For Windows PowerShell:**
```powershell
$env:HF_TOKEN="your_groq_api_key"
$env:API_BASE_URL="https://api.groq.com/openai/v1"
$env:MODEL_NAME="llama-3.1-8b-instant"

python inference.py
```

```

### requirements.txt
```txt
openenv-core
fastapi
uvicorn[standard]
websockets
pydantic
openai

```

### scratch\codebase_extraction.md
```markdown

```

### scratch\codebase_extraction_utf8.md
```markdown
Error reading file: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

### scratch\extract_codebase.py
```python
import os

def extract_codebase(root_dir):
    exclusions = {'.git', '__pycache__', '.venv', 'venv'}
    excluded_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz'}
    included_extensions = {'.py', '.yaml', '.toml', '.txt', '.md', 'Dockerfile'}
    
    # 1. Directory Tree
    print("# Repository Architecture")
    print("```text")
    for root, dirs, files in os.walk(root_dir):
        # In-place modify dirs to skip excluded ones
        dirs[:] = [d for d in dirs if d not in exclusions]
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if not any(f.endswith(ext) for ext in excluded_extensions) and f != 'uv.lock':
                print(f"{sub_indent}{f}")
    print("```\n")

    # 2. File Contents
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclusions]
        for f in files:
            if f == 'uv.lock':
                continue
            
            ext = os.path.splitext(f)[1].lower()
            if any(f == name for name in included_extensions) or ext in included_extensions:
                file_path = os.path.join(root, f)
                rel_path = os.path.relpath(file_path, root_dir)
                
                print(f"### {rel_path}")
                
                lang = ext[1:] if ext else ""
                if f == 'Dockerfile':
                    lang = "dockerfile"
                elif ext == ".py":
                    lang = "python"
                elif ext == ".md":
                    lang = "markdown"
                
                print(f"```{lang}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        print(file.read())
                except Exception as e:
                    print(f"Error reading file: {e}")
                print("```\n")

if __name__ == "__main__":
    import sys
    output_file = "codebase_extraction.md"
    with open(output_file, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            extract_codebase("g:\\hj\\meta\\cloud_infra_env")
        finally:
            sys.stdout = original_stdout
    print(f"Extraction completed successfully to {output_file}")

```

### scratch\verify_clamp.py
```python

def clamp_score(score):
    """Guarantees score is strictly between 0 and 1 by clamping to [0.01, 0.99]"""
    try:
        val = float(score)
        return max(0.01, min(0.99, val))
    except (ValueError, TypeError):
        return 0.01

test_cases = [
    (0.0, "0.01"),
    (1.0, "0.99"),
    (0.999, "0.99"),
    (0.001, "0.01"),
    (0.5, "0.50"),
    (0.99, "0.99"),
    (0.01, "0.01"),
    (-1.0, "0.01"),
    (2.0, "0.99"),
    ("invalid", "0.01"),
    (None, "0.01")
]

print("Testing clamp_score and formatting:")
for val, expected in test_cases:
    clamped = clamp_score(val)
    formatted = f"{clamped:.2f}"
    status = "PASS" if formatted == expected else "FAIL"
    print(f"Input: {val} -> Clamped: {clamped} -> Formatted: {formatted} (Expected: {expected}) -> {status}")

```

### server\app.py
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server import create_fastapi_app
from server.environment import CloudEnvironment
from models import CloudAction, CloudObservation
import uvicorn

app = create_fastapi_app(CloudEnvironment, CloudAction, CloudObservation)

@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv Cloud Admin Simulator is running"}

def main():
    #uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

```

### server\Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

```

### server\environment.py
```python
import random
import uuid
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openenv.core.env_server import Environment
from models import CloudAction, CloudObservation, CloudState

class CloudEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = CloudState()

    def reset(self, seed=None, episode_id=None, difficulty=None, **kwargs) -> CloudObservation:
        episode_id = episode_id or str(uuid.uuid4())
        
        if not difficulty:
            difficulty = random.choice(["easy", "medium", "hard"])
            
        self._state = CloudState(
            episode_id=episode_id,
            step_count=0,
            difficulty=difficulty,
            max_steps=15,
            resources={},
            users={}
        )
        
        self._state.resources["i-001"] = {"id": "i-001", "type": "instance", "status": "running", "tags": {"env": "prod"}, "creator": "admin"}
        self._state.resources["i-002"] = {"id": "i-002", "type": "instance", "status": "running", "tags": {"env": "dev"}, "creator": "dev_user"}
        self._state.resources["b-001"] = {"id": "b-001", "type": "bucket", "name": "app-logs", "public_access": False}
        
        self._state.users["admin"] = {"id": "admin", "status": "active"}
        self._state.users["dev_user"] = {"id": "dev_user", "status": "active"}

        if difficulty == "easy":
            self._state.resources["i-temp"] = {"id": "i-temp", "type": "instance", "status": "running", "tags": {"purpose": "temporary"}, "creator": "dev_user"}
            message = "Task: We have an instance tagged with 'purpose': 'temporary' that was left running. Find it and STOP it."
        
        elif difficulty == "medium":
            self._state.resources["b-002"] = {"id": "b-002", "type": "bucket", "name": "public-assets", "public_access": True}
            message = "Task: One of our storage buckets named 'public-assets' has public_access set to True. Update it to be private (public_access=False)."
        
        else: 
            self._state.users["hacker123"] = {"id": "hacker123", "status": "active"}
            self._state.resources["i-hack"] = {"id": "i-hack", "type": "instance", "status": "running", "tags": {"env": "unknown"}, "creator": "hacker123"}
            message = "Task: The user 'hacker123' is compromised. DISABLE the user, find the instance created by this user, and TERMINATE the instance."
            
        message += "\nAvailable commands: \n- LIST_INSTANCES\n- STOP_INSTANCE (target_id=<id>)\n- TERMINATE_INSTANCE (target_id=<id>)\n- LIST_BUCKETS\n- UPDATE_BUCKET_ACCESS (target_id=<id>, args='private' or 'public')\n- LIST_USERS\n- DISABLE_USER (target_id=<id>)\n- DONE (Call when you believe the task is fully complete)"
        
        return CloudObservation(
            done=False,
            reward=0.01, # FIXED: Strictly greater than 0
            message=message,
            outputs=None
        )

    def step(self, action: CloudAction, timeout_s=None, **kwargs) -> CloudObservation:
        self._state.step_count += 1
        
        command = action.command.upper().strip()
        target_id = action.target_id
        args = action.args
        message = f"Command {command} executed successfully."
        outputs = None
        done = False
        reward = 0.01 # FIXED: Baseline reward avoids 0.0
        
        if self._state.step_count >= self._state.max_steps:
             done = True
             message = "Maximum steps reached."
             reward = self._calculate_reward()
             return CloudObservation(done=done, reward=reward, message=message, outputs=outputs)
             
        try:
            if command == "LIST_INSTANCES":
                instances = [r for r in self._state.resources.values() if r["type"] == "instance"]
                outputs = instances
            
            elif command == "STOP_INSTANCE":
                if target_id in self._state.resources and self._state.resources[target_id]["type"] == "instance":
                    self._state.resources[target_id]["status"] = "stopped"
                else:
                    message = f"Error: Instance {target_id} not found."
                    
            elif command == "TERMINATE_INSTANCE":
                if target_id in self._state.resources and self._state.resources[target_id]["type"] == "instance":
                    self._state.resources[target_id]["status"] = "terminated"
                else:
                    message = f"Error: Instance {target_id} not found."
            
            elif command == "LIST_BUCKETS":
                buckets = [r for r in self._state.resources.values() if r["type"] == "bucket"]
                outputs = buckets
                
            elif command == "UPDATE_BUCKET_ACCESS":
                if target_id in self._state.resources and self._state.resources[target_id]["type"] == "bucket":
                    if args and args.lower() == "private":
                        self._state.resources[target_id]["public_access"] = False
                    elif args and args.lower() == "public":
                        self._state.resources[target_id]["public_access"] = True
                    else:
                        message = "Error: Invalid args for UPDATE_BUCKET_ACCESS. Use 'private' or 'public'."
                else:
                    message = f"Error: Bucket {target_id} not found."
                    
            elif command == "LIST_USERS":
                users = list(self._state.users.values())
                outputs = users
                
            elif command == "DISABLE_USER":
                if target_id in self._state.users:
                    self._state.users[target_id]["status"] = "disabled"
                else:
                    message = f"Error: User {target_id} not found."
                    
            elif command == "DONE":
                done = True
                reward = self._calculate_reward()
                if reward >= 0.9: # FIXED: Adjusted for new maximum score
                    message = "Task completed successfully! Perfect score."
                elif reward > 0.1:
                    message = f"Task partially completed. Score: {reward}"
                else:
                    message = "Task failed."
            else:
                message = f"Error: Unknown command {command}."
        except Exception as e:
            message = f"Error executing command: {str(e)}"
            
        return CloudObservation(
            done=done,
            reward=reward if done else 0.01, # FIXED: Avoids 0.0
            message=message,
            outputs=outputs
        )

    def _calculate_reward(self) -> float:
        """Mathematically bounds the reward strictly between 0.01 and 0.99"""
        score = 0.1 # Base score
        
        if self._state.difficulty == "easy":
            temp_instances = [r for r in self._state.resources.values() if r.get("tags", {}).get("purpose") == "temporary"]
            if temp_instances and temp_instances[0]["status"] in ["stopped", "terminated"]:
                score += 0.8
                
        elif self._state.difficulty == "medium":
            bucket = next((r for r in self._state.resources.values() if r.get("name") == "public-assets"), None)
            if bucket and bucket["public_access"] == False:
                score += 0.8
                
        elif self._state.difficulty == "hard":
            u_disabled = self._state.users.get("hacker123", {}).get("status") == "disabled"
            hack_inst = next((r for r in self._state.resources.values() if r.get("creator") == "hacker123"), None)
            i_terminated = hack_inst and hack_inst["status"] == "terminated"
            
            if u_disabled and i_terminated:
                score += 0.8
            elif u_disabled or i_terminated:
                score += 0.4
                
        # DOUBLE CLAMP: Ensure it NEVER hits 0.0 or 1.0
        return float(max(0.01, min(0.99, score)))

    @property
    def state(self) -> CloudState:
        return self._state
```

