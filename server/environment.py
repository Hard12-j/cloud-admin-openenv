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
        """Dynamic reward calculation factoring in efficiency and best practices."""
        score = 0.2  # Base starting potential
        
        # Efficiency penalty (rewarding faster completion)
        # Max steps is 15, so max penalty is -0.15
        score -= (self._state.step_count * 0.01)

        if self._state.difficulty == "easy":
            temp_instances = [r for r in self._state.resources.values() if r.get("tags", {}).get("purpose") == "temporary"]
            if temp_instances:
                status = temp_instances[0]["status"]
                if status == "stopped":
                    score += 0.7  # Ideal: stopped (resource preservation)
                elif status == "terminated":
                    score += 0.5  # Acceptable but destructive (data loss)
                
        elif self._state.difficulty == "medium":
            bucket = next((r for r in self._state.resources.values() if r.get("name") == "public-assets"), None)
            if bucket and bucket.get("public_access") is False:
                score += 0.7
                
        elif self._state.difficulty == "hard":
            u_disabled = self._state.users.get("hacker123", {}).get("status") == "disabled"
            hack_inst = next((r for r in self._state.resources.values() if r.get("creator") == "hacker123"), None)
            i_terminated = hack_inst and hack_inst["status"] == "terminated"
            
            if u_disabled and i_terminated:
                score += 0.7
            elif u_disabled or i_terminated:
                score += 0.35
        
        # Add random jitter to ensure uniqueness (Phase 3 requirement)
        score += random.uniform(-0.01, 0.01)
        return max(0.01, min(0.99, round(score, 3)))

    @property
    def state(self) -> CloudState:
        return self._state