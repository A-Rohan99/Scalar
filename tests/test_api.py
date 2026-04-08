import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from app.api import app

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["version"] == "1.1.0"
        assert "active_sessions" in response.json()

@pytest.mark.asyncio
async def test_reset():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/reset", json={"task_id": 1, "seed": 42})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "observation" in data
        assert data["observation"]["task_id"] == 1

@pytest.mark.asyncio
async def test_step_after_reset():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        reset_resp = await ac.post("/reset", json={"task_id": 1, "seed": 42})
        session_id = reset_resp.json()["session_id"]
        
        response = await ac.post("/step", headers={"X-Session-ID": session_id}, json={
            "action_type": "query",
            "sql": "SELECT * from sqlite_master"
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "reward" in data
        assert "observation" in data

@pytest.mark.asyncio
async def test_grader_without_reset():
    from app.api import sessions, sessions_lock
    async with sessions_lock:
        sessions.clear()
        
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/grader", headers={"X-Session-ID": "non-existent-session-id"})
        assert response.status_code == 400

@pytest.mark.asyncio
async def test_tasks_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3
        assert "action_schema" in data

@pytest.mark.asyncio
async def test_baseline_nonblocking():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/baseline")
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "running"
        assert "poll_url" in data
        
        job_id = data["job_id"]
        
        health_resp = await ac.get("/health")
        assert health_resp.status_code == 200
        
        for _ in range(20):
            poll_resp = await ac.get(f"/baseline/{job_id}")
            assert poll_resp.status_code == 200
            poll_data = poll_resp.json()
            if poll_data["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.5)
            
        assert poll_data["status"] in ("done", "error")

@pytest.mark.asyncio
async def test_session_eviction():
    from app.api import sessions, sessions_lock, reset_limiter
    old_max = reset_limiter.max_calls
    reset_limiter.max_calls = 100
    try:
        async with sessions_lock:
            sessions.clear()
            
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            for i in range(55):
                await ac.post("/reset", json={"task_id": 1, "seed": i})
                
            health_resp = await ac.get("/health")
            assert health_resp.status_code == 200
            data = health_resp.json()
            assert data["active_sessions"] == 50
    finally:
        reset_limiter.max_calls = old_max

