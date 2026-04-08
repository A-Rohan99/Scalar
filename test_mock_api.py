import httpx
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

print("1. Call /reset without session ID")
r1 = client.post("/reset", json={"task_id": 1, "seed": 42})
print("Reset response:", r1.status_code, r1.text)
sid = r1.json().get("session_id")
print("Session ID:", sid)

print("2. Call /grader with session ID")
r2 = client.get("/grader", headers={"X-Session-ID": sid})
print("Grader response:", r2.status_code, r2.text)
