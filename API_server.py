import nest_asyncio, threading, uuid
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
from backend.train import train_model
import uvicorn
nest_asyncio.apply()

app = FastAPI()
jobs = {}

class TrainReq(BaseModel):
    epochs: int = 5
    lr: float = 1e-4

@app.post("/train")
def train(req: TrainReq):
    jid = str(uuid.uuid4())
    jobs[jid] = {"state": "running"}
    threading.Thread(target=_wrap_train, args=(jid, req), daemon=True).start()
    return {"job_id": jid}

@app.get("/status/{jid}")
def status(jid: str):
    return jobs.get(jid, {"state": "unknown"})

def _wrap_train(jid, req):
    try:
        train_model(req.epochs, req.lr)
        jobs[jid] = {"state": "completed"}
    except Exception as e:
        jobs[jid] = {"state": "error", "msg": str(e)}

# --- ngrok で公開 ---
public_url = ngrok.connect(8000).public_url
print("✔ API endpoint:", public_url)

uvicorn.run(app, host="0.0.0.0", port=8000)
