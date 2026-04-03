# Cloud Infrastructure Manager (OpenEnv)

## Description
This is a standard OpenEnv-based project simulating an AWS/GCP administrative task. The agent has tools to list instances, buckets, users, and modify their states in order to resolve configuration and security tasks (like shutting down unwanted instances, disabling compromised users, and changing public bucket accesses).

## Action Space
`CloudAction(command=str, target_id=str, args=str)`
* `command` can be one of: `LIST_INSTANCES`, `STOP_INSTANCE`, `TERMINATE_INSTANCE`, `LIST_BUCKETS`, `UPDATE_BUCKET_ACCESS`, `LIST_USERS`, `DISABLE_USER`, `DONE`.

## Observation Space
`CloudObservation(message=str, outputs=List[dict])`

## Tasks Graded
* **Easy:** Stop a temporary instance based on its tags. (Reward 1.0)
* **Medium:** Ensure an S3 bucket named 'public-assets' has public access disabled. (Reward 1.0)
* **Hard:** Disable a compromised user and terminate their associated instance. (Reward: 0.5 partial, 1.0 full)

## Setup & Run
This environment requires `openenv-core`.

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## AI Inference Baseline
Run the baseline AI agent that interacts with the app locally:
```cmd
set HF_TOKEN=your_hugging_face_token
set API_BASE_URL=https://api-inference.huggingface.co/v1/
set MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct

python inference.py
```
