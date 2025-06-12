# api_server.py  ── FastAPI + ngrok だけに集中
import uuid, threading, nest_asyncio, uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
from backend.train import train_model   # ← backend/train.py が前提
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
    threading.Thread(target=_wrap, args=(jid, req), daemon=True).start()
    return {"job_id": jid}

@app.get("/status/{jid}")
def status(jid: str):
    return jobs.get(jid, {"state": "unknown"})

def _wrap(jid, req):
    try:
        train_model(req.epochs, req.lr)
        jobs[jid] = {"state": "completed"}
    except Exception as exc:
        jobs[jid] = {"state": "error", "msg": str(exc)}

if __name__ == "__main__":
    # ❶ ngrok トンネル：ここ“だけ”で 1 回張る
    public_url = ngrok.connect(8000).public_url
    print("✔ API endpoint:", public_url)

    # ❷ FastAPI 起動（ポートは ngrok と同じ 8000）
    uvicorn.run(app, host="0.0.0.0", port=8000)
