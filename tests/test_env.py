import pytest
from app.env import DataOpsEnv
from app.models import QueryAction, DDLAction

@pytest.mark.asyncio
async def test_reset_returns_observation():
    env = DataOpsEnv()
    obs = await env.reset(task_id=1, seed=42)
    assert obs.current_step == 0
    assert obs.max_steps > 0
    assert obs.task_id == 1
    assert "description" in obs.task_description or "Find" in obs.task_description
    assert obs.schema_info

@pytest.mark.asyncio
async def test_step_returns_reward():
    env = DataOpsEnv()
    await env.reset(task_id=1, seed=42)
    action = QueryAction(action_type="query", sql="SELECT 1")
    obs, reward = await env.step(action)
    assert -1.0 <= reward.step_reward <= 1.0
    assert obs.current_step == 1

@pytest.mark.asyncio
async def test_different_seeds_differ():
    env1 = DataOpsEnv()
    obs1 = await env1.reset(task_id=1, seed=42)
    
    env2 = DataOpsEnv()
    obs2 = await env2.reset(task_id=1, seed=99)
    
    assert list(obs1.schema_info.keys()) != list(obs2.schema_info.keys())

@pytest.mark.asyncio
async def test_truncation():
    env = DataOpsEnv()
    await env.reset(task_id=1, seed=42)
    env.state.max_steps = 3
    
    action = QueryAction(action_type="query", sql="SELECT 1")
    await env.step(action)
    await env.step(action)
    obs, reward = await env.step(action)
    
    assert reward.truncated is True
    assert reward.done is True

@pytest.mark.asyncio
async def test_no_hardcoding():
    table_names = set()
    for i in range(10):
        env = DataOpsEnv()
        await env.reset(task_id=1, seed=100+i)
        main_table = env.state.table_registry["main"]
        table_names.add(main_table)
    assert len(table_names) == 10

@pytest.mark.asyncio
async def test_sql_injection_blocked():
    env = DataOpsEnv()
    await env.reset(task_id=1, seed=42)
    step_before = env.state.current_step
    
    action = DDLAction(action_type="ddl", sql="DROP TABLE sqlite_master")
    obs, reward = await env.step(action)
    
    assert obs.last_action_status == "ERROR"
    assert "blocked" in obs.last_error_message.lower()
    assert env.state.current_step == step_before  # Step count did not increment
    
@pytest.mark.asyncio
async def test_sql_valid_ddl_allowed():
    env = DataOpsEnv()
    await env.reset(task_id=1, seed=42)
    step_before = env.state.current_step
    
    main_table = env.state.table_registry["main"]
    col_name = env.state.column_registry["name"]
    action = DDLAction(action_type="ddl", sql=f"UPDATE {main_table} SET {col_name}='fixed'")
    obs, reward = await env.step(action)
    
    assert obs.last_action_status == "SUCCESS"
    assert env.state.current_step == step_before + 1

@pytest.mark.asyncio
async def test_sql_sqlite_master_write_blocked():
    env = DataOpsEnv()
    await env.reset(task_id=1, seed=42)
    step_before = env.state.current_step
    
    action = DDLAction(action_type="ddl", sql="DELETE FROM sqlite_master WHERE name='x'")
    obs, reward = await env.step(action)
    
    assert obs.last_action_status == "ERROR"
    assert "sqlite_master" in obs.last_error_message.lower()
    assert env.state.current_step == step_before

def test_exception_sql_trigger_returns_400_or_error_obs():
    import uuid
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    
    res = client.post("/reset", json={"task_id": 1})
    assert res.status_code == 200
    sid = res.json()["session_id"]
    
    res = client.post("/step", json={"action_type": "ddl", "sql": "CREATE TRIGGER t AFTER INSERT ON nonexistent BEGIN SELECT 1; END"}, headers={"X-Session-ID": sid})
    if res.status_code == 500:
        print("500 ERROR TEXT:", res.text)
    assert res.status_code in [200, 400], f"Trigger test failed with {res.status_code}"
    # Ensure it's not a 500 traceback
    assert res.status_code != 500

def test_exception_pragma_info_dropped_view():
    import uuid
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    
    res = client.post("/reset", json={"task_id": 1})
    sid = res.json()["session_id"]
    client.post("/step", json={"action_type": "ddl", "sql": "CREATE TABLE ttt (id INT)"}, headers={"X-Session-ID": sid})
    client.post("/step", json={"action_type": "ddl", "sql": "CREATE VIEW v2 AS SELECT * FROM ttt"}, headers={"X-Session-ID": sid})
    client.post("/step", json={"action_type": "ddl", "sql": "DROP TABLE ttt"}, headers={"X-Session-ID": sid})
    
    res = client.post("/step", json={"action_type": "query", "sql": "PRAGMA table_info(v2)"}, headers={"X-Session-ID": sid})
    # Must not crash, should return error observation or 400
    assert res.status_code != 500
    if res.status_code == 200:
        assert isinstance(res.json().get("observation"), dict)

def test_exception_invalid_seed():
    from fastapi.testclient import TestClient
    from app.api import app
    client = TestClient(app)
    
    res = client.post("/reset", json={"task_id": 1, "seed": "not_an_int"})
    assert res.status_code == 422 # Pydantic validation error
