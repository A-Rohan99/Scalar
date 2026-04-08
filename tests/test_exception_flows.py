import pytest
import httpx
import uuid
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_exception_sql_trigger_returns_observation():
    # 1. Reset
    res = client.post("/reset", json={"task_id": 1})
    assert res.status_code == 200
    sid = res.json()["session_id"]
    
    # Send complex trigger
    res = client.post("/step", json={"action_type": "ddl", "sql": "CREATE TRIGGER t AFTER INSERT ON nonexistent BEGIN SELECT 1; END"}, headers={"X-Session-ID": sid})
    
    # Did it return 200 with error observation, or 400?
    print("Trigger result:", res.status_code, res.json())

def test_exception_pragma_info_dropped_view():
    res = client.post("/reset", json={"task_id": 1})
    sid = res.json()["session_id"]
    
    # CREATE VIEW
    res = client.post("/step", json={"action_type": "ddl", "sql": "CREATE VIEW v AS SELECT 1"}, headers={"X-Session-ID": sid})
    
    # DROP UNDERLYING ? (Wait, just drop the view instead of dropping the table) 
    # Actually wait. Just drop table and run pragma on view.
    client.post("/step", json={"action_type": "ddl", "sql": "CREATE TABLE ttt (id INT)"}, headers={"X-Session-ID": sid})
    client.post("/step", json={"action_type": "ddl", "sql": "CREATE VIEW v2 AS SELECT * FROM ttt"}, headers={"X-Session-ID": sid})
    client.post("/step", json={"action_type": "ddl", "sql": "DROP TABLE ttt"}, headers={"X-Session-ID": sid})
    
    # Query pragma on broken view
    res = client.post("/step", json={"action_type": "query", "sql": "PRAGMA table_info(v2)"}, headers={"X-Session-ID": sid})
    print("Pragma result:", res.status_code, res.json())

def test_exception_invalid_seed():
    # Invalid type for seed should throw 422
    res = client.post("/reset", json={"task_id": 1, "seed": "not_an_int"})
    print("Invalid seed reset:", res.status_code, res.json())
