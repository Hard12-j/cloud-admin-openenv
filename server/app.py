import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openenv.core.env_server import create_fastapi_app
from .environment import CloudEnvironment
from models import CloudAction, CloudObservation

app = create_fastapi_app(CloudEnvironment, CloudAction, CloudObservation)
