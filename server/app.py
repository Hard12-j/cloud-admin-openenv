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
    uvicorn.run("server.app:app", host="[IP_ADDRESS]", port=port)

if __name__ == "__main__":
    main()
