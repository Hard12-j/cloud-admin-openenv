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
