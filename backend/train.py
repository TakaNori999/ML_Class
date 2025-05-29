# backend/train.py  ※とりあえず動くダミー
import time

def train_model(epochs: int = 5, lr: float = 1e-4):
    for epoch in range(epochs):
        time.sleep(2)
        print(f"[{epoch+1}/{epochs}] lr={lr} loss=0.{epoch}{epoch}")
