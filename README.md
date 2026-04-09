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
