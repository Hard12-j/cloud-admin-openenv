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
        reward = payload.get("reward", 0.0)
        
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
