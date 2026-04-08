import os
import re
import uuid
import asyncio
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from collections import Counter, defaultdict, deque
import time

from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dataclasses import dataclass

@dataclass
class LeaderboardEntry:
    model_name: str
    task_id: int
    score: float
    steps_taken: int
    timestamp: str
    session_id: str

leaderboard: List[LeaderboardEntry] = []

from app.env import DataOpsEnv
from app.models import Action, Observation, StateSnapshot, CompletedEpisode
from app.tasks import TASK_REGISTRY, get_action_schema

class SimpleRateLimiter:
    """
    Sliding window rate limiter using in-memory deques.
    No Redis, no external dependencies, works in single-process uvicorn.
    """
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)"""
        async with self._lock:
            now = time.time()
            window_start = now - self.window
            
            # Remove calls outside the window
            calls = self._calls[key]
            while calls and calls[0] < window_start:
                calls.popleft()
            
            if len(calls) >= self.max_calls:
                retry_after = int(calls[0] + self.window - now) + 1
                return False, retry_after
            
            calls.append(now)
            return True, 0

# Instantiate: 10 resets per minute per IP
reset_limiter = SimpleRateLimiter(max_calls=10, window_seconds=60)
# Baseline runs are expensive: 2 per hour per IP
baseline_limiter = SimpleRateLimiter(max_calls=2, window_seconds=3600)

app = FastAPI(title="OpenDataOpsEnv API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, DataOpsEnv] = {}
sessions_lock = asyncio.Lock()
baseline_jobs: Dict[str, dict] = {}

completed_episodes: List[CompletedEpisode] = []
global_stats_lock = asyncio.Lock()

async def session_cleanup_task():
    while True:
        await asyncio.sleep(300)
        
        # PHASE 1: Identify stale keys WITHOUT holding the lock
        async with sessions_lock:
            current_keys = list(sessions.keys())
            
        # PHASE 2: Check staleness outside the lock
        now = datetime.now(timezone.utc)
        stale_keys = []
        for sid in current_keys:
            env = sessions.get(sid)
            if env and (now - env.last_activity).total_seconds() > 1800:
                stale_keys.append(sid)
                
        # PHASE 3: Delete only the stale keys, re-acquire lock briefly
        if stale_keys:
            async with sessions_lock:
                for sid in stale_keys:
                    sessions.pop(sid, None)
            print(f"Cleaned up {len(stale_keys)} stale sessions")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"UNHANDLED: {request.url} — {traceback.format_exc()[:300]}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "endpoint": str(request.url.path),
            "message": "The environment encountered an unexpected error. The episode has been preserved.",
            "action": "Call GET /state to check current episode status, or POST /reset to start fresh."
        }
    )

import sqlite3
@app.exception_handler(sqlite3.Error)
async def sqlite_exception_handler(request: Request, exc: sqlite3.Error):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Database error",
            "message": str(exc),
            "last_action_status": "ERROR"
        }
    )

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(session_cleanup_task())
    
    now_str = datetime.now(timezone.utc).isoformat()
    baselines = [
        LeaderboardEntry("gpt-4o", 1, 0.97, 5, now_str, str(uuid.uuid4())),
        LeaderboardEntry("gpt-4o-mini", 1, 0.82, 6, now_str, str(uuid.uuid4())),
        LeaderboardEntry("gpt-4o-mini", 2, 0.61, 10, now_str, str(uuid.uuid4())),
        LeaderboardEntry("gpt-4o-mini", 3, 0.34, 15, now_str, str(uuid.uuid4()))
    ]
    leaderboard.extend(baselines)
    
    print("OpenDataOpsEnv ready on port 7860")

async def get_session(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")) -> tuple[str, DataOpsEnv]:
    session_id = x_session_id
    async with sessions_lock:
        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = DataOpsEnv()
        return session_id, sessions[session_id]

class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=4, description="Task to initialise. Defaults to 1 if not provided.")
    seed: Optional[int] = Field(default=None, description="Random seed. Random if not provided.")
    difficulty_multiplier: float = Field(default=1.0, ge=0.5, le=2.0)

@app.get("/", response_class=HTMLResponse, description="Landing page for HF Spaces")
async def root():
    tasks_html = ""
    for task in TASK_REGISTRY.values():
        diff_class = str(task['difficulty']).lower()
        tasks_html += f"""
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #eee;">{task['id']}</td>
            <td style="padding: 12px; border-bottom: 1px solid #eee;"><strong>{task['name']}</strong></td>
            <td style="padding: 12px; border-bottom: 1px solid #eee;"><span class="badge {diff_class}">{str(task['difficulty']).upper()}</span></td>
            <td style="padding: 12px; border-bottom: 1px solid #eee;">{task['description']}</td>
        </tr>
        """
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OpenDataOpsEnv v1.1.0</title>
        <style>
            :root {{
                --primary: #2563eb;
                --text: #1f2937;
                --bg: #f8fafc;
                --surface: #ffffff;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: var(--text);
                background-color: var(--bg);
                max-width: 1000px;
                margin: 0 auto;
                padding: 2rem;
            }}
            .card {{
                background: var(--surface);
                border-radius: 8px;
                padding: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }}
            h1 {{ color: var(--primary); margin-top: 0; }}
            h2 {{ color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
            .guarantee {{
                background-color: #dcfce7;
                border-left: 4px solid #22c55e;
                padding: 1rem;
                border-radius: 0 4px 4px 0;
                font-weight: 500;
            }}
            table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
            th {{ background-color: #f1f5f9; text-align: left; padding: 12px; }}
            .badge {{
                padding: 4px 8px;
                border-radius: 9999px;
                font-size: 0.8rem;
                font-weight: bold;
            }}
            .easy {{ background-color: #dcfce7; color: #166534; }}
            .medium {{ background-color: #fef08a; color: #9a3412; }}
            .hard {{ background-color: #fee2e2; color: #991b1b; }}
            pre {{
                background-color: #1e293b;
                color: #f8fafc;
                padding: 1rem;
                border-radius: 6px;
                overflow-x: auto;
            }}
            a {{ color: var(--primary); text-decoration: none; font-weight: 500; }}
            a:hover {{ text-decoration: underline; }}
            .nav-links {{ display: flex; gap: 1.5rem; margin-top: 1rem; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>OpenDataOpsEnv <span>v1.1.0</span></h1>
            <p>Welcome to the <strong>DataOps incident-response environment</strong>. This sandbox simulates realistic database pipeline failures, PII masking tasks, and data cleaning operations. Agents connect to dynamically seeded SQLite states, execute SQL commands to surgically diagnose and repair the infrastructure, and receive dense reward signals mapping directly back to underlying grader validations.</p>
            
            <div class="guarantee">
                No-Hardcoding Guarantee: Every single episode dynamically generates unique randomly-seeded table names, columns, and data points strictly ensuring agents cannot memorize schemas.
            </div>

            <div class="nav-links">
                <a href="/docs">📚 API Documentation (Swagger)</a>
                <a href="/tasks">📋 View Raw Tasks JSON</a>
                <a href="/state">🔍 Current State</a>
                <a href="/leaderboard">🏆 Leaderboard</a>
            </div>
        </div>

        <div class="card">
            <h2>Available Incident Tasks</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Task Name</th>
                        <th>Difficulty</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {tasks_html}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>Try it via cURL</h2>
            <p>Instantiate a dynamic environment locally returning an isolated session trace:</p>
            <pre><code>curl -X POST http://localhost:7860/reset \\
     -H "Content-Type: application/json" \\
     -d '{{"task_id": 1, "seed": 42}}'</code></pre>
            <br>
            <p>Perform a step sending an action within the isolated session:</p>
            <pre><code>curl -X POST http://localhost:7860/step \\
     -H "Content-Type: application/json" \\
     -H "X-Session-ID: &lt;your-session-id&gt;" \\
     -d '{{"action_type": "query", "sql": "SELECT name FROM sqlite_master"}}'</code></pre>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health", description="Health check endpoint")
def health():
    return {"status": "ok", "version": "1.1.0", "active_sessions": len(sessions)}

@app.get("/stats", description="Get aggregate statistics across all completed episodes")
async def get_stats():
    async with global_stats_lock:
        if not completed_episodes:
            return {
                "total_episodes": 0,
                "by_task": {},
                "most_common_failure_actions": [],
                "mean_episode_length": 0.0
            }
            
        total = len(completed_episodes)
        total_steps_all = sum(ep.total_steps for ep in completed_episodes)
        mean_episode_length = round(total_steps_all / total, 2)
        
        by_task = {}
        all_failed_actions = []
        
        for task_id in [1, 2, 3]:
            eps = [ep for ep in completed_episodes if ep.task_id == task_id]
            if not eps:
                continue
                
            task_count = len(eps)
            mean_score = sum(ep.final_score for ep in eps) / task_count
            mean_steps = sum(ep.total_steps for ep in eps) / task_count
            perfect = sum(1 for ep in eps if ep.final_score >= 0.99)
            
            by_task[str(task_id)] = {
                "count": task_count,
                "mean_score": round(mean_score, 2),
                "mean_steps": round(mean_steps, 2),
                "perfect_scores": perfect
            }
            
        for ep in completed_episodes:
            all_failed_actions.extend(ep.failed_actions)
            
        counter = Counter(all_failed_actions)
        most_common = [act for act, count in counter.most_common(5)]
        
        return {
            "total_episodes": total,
            "by_task": by_task,
            "most_common_failure_actions": most_common,
            "mean_episode_length": mean_episode_length
        }


@app.post("/reset", description="Reset the environment")
async def reset_env(request: Request, req: ResetRequest = None, x_session_id: Optional[str] = Header(None, alias="X-Session-ID")):
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = await reset_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum 10 resets per minute. Retry after {retry_after} seconds.",
                "retry_after": retry_after
            }
        )

    if req is None:
        req = ResetRequest()
    session_id = x_session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        
    async with sessions_lock:
        if len(sessions) >= 50:
            oldest_sid = min(sessions.keys(), key=lambda k: sessions[k].last_activity)
            del sessions[oldest_sid]
            
        new_env = DataOpsEnv()
        sessions[session_id] = new_env
        
    obs = await new_env.reset(req.task_id, req.seed, req.difficulty_multiplier)
    return {"session_id": session_id, "observation": obs}

@app.post("/step", description="Take a step in the environment")
async def step_env(action: Action, session: tuple = Depends(get_session), x_model_name: Optional[str] = Header("anonymous", alias="X-Model-Name")):
    session_id, env = session
    if not env.state or env.state.done:
        raise HTTPException(status_code=400, detail="Episode not active")
    
    obs, reward = await env.step(action, session_id)
    
    if reward.done:
        failed_acts = []
        for t_item in env.state.trajectory:
            t_obs = t_item.get("observation", {})
            t_act = t_item.get("action", {})
            if t_obs.get("last_action_status") == "ERROR":
                sql = t_act.get("sql", "").strip().upper()
                if sql:
                    failed_acts.append(" ".join(sql.split()[:2]))

        comp_ep = CompletedEpisode(
            episode_id=env.state.episode_id,
            task_id=env.state.task_id,
            total_steps=env.state.current_step,
            final_score=reward.grader_score_after,
            failed_actions=failed_acts
        )
        async with global_stats_lock:
            completed_episodes.append(comp_ep)
            
        entry = LeaderboardEntry(
            model_name=str(x_model_name) if x_model_name else "anonymous",
            task_id=env.state.task_id,
            score=reward.grader_score_after,
            steps_taken=env.state.current_step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id
        )
        leaderboard.append(entry)
        leaderboard.sort(key=lambda x: (-x.score, x.steps_taken))
        
        task_entries = [e for e in leaderboard if e.task_id == env.state.task_id][:100]
        other_entries = [e for e in leaderboard if e.task_id != env.state.task_id]
        leaderboard.clear()
        leaderboard.extend(other_entries + task_entries)
        leaderboard.sort(key=lambda x: (-x.score, x.steps_taken))

    return {
        "session_id": session_id,
        "observation": obs,
        "reward": reward.step_reward,
        "done": reward.done,
        "truncated": reward.truncated,
        "info": {
            "reward_breakdown": reward.reward_breakdown,
            "grader_score": env.grader_score(),
            "grader_score_before": reward.grader_score_before,
            "grader_score_after": reward.grader_score_after
        }
    }

@app.get("/leaderboard", description="View the top model performances")
async def get_leaderboard():
    board_response = {"task_1": [], "task_2": [], "task_3": []}
    
    for task_id in [1, 2, 3]:
        entries = [e for e in leaderboard if e.task_id == task_id]
        entries.sort(key=lambda x: (-x.score, x.steps_taken))
        
        for i, entry in enumerate(entries[:100]):
            board_response[f"task_{task_id}"].append({
                "rank": i + 1,
                "model": entry.model_name,
                "score": round(entry.score, 2),
                "steps": entry.steps_taken,
                "timestamp": entry.timestamp,
                "session_id": entry.session_id
            })
            
    async with global_stats_lock:
        tot_eps = len(completed_episodes)
            
    return {
        "leaderboard": board_response,
        "total_episodes_recorded": max(tot_eps, sum(len(lst) for lst in board_response.values())),
        "environment_version": "1.1.0"
    }

@app.get("/state", description="Get current state snapshot")
async def get_state(session: tuple = Depends(get_session)):
    session_id, env = session
    if not env.state:
        raise HTTPException(status_code=400, detail="No active episode")
    try:
        return {"session_id": session_id, "state": env.get_state()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/grader", description="Get current grader score")
async def get_grader(x_session_id: Optional[str] = Header(None, alias="X-Session-ID")):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")
    async with sessions_lock:
        env = sessions.get(x_session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{x_session_id}' not found. Call POST /reset first.")
    if not env.state:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset to start one.")
    return {
        "session_id": x_session_id,
        "task_id": env.state.task_id,
        "score": env.grader_score(),
        "step": env.state.current_step,
        "done": env.state.done
    }

@app.get("/tasks", description="List all tasks and action schema")
def get_tasks():
    return {
        "tasks": list(TASK_REGISTRY.values()),
        "action_schema": get_action_schema()
    }

async def _run_baseline_job(job_id: str):
    job = baseline_jobs[job_id]
    env_vars = os.environ.copy()
    
    try:
        process = await asyncio.create_subprocess_exec(
            "python", "baseline/inference.py",
            env=env_vars,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        async def read_stream(stream, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded_line = line.decode('utf-8', errors='replace')
                job["log"] += decoded_line
                
                if not is_stderr:
                    match = re.search(r"SCORE\s+task_(\d+):\s*([\d\.]+)", decoded_line, re.IGNORECASE)
                    if match:
                        task_num = match.group(1)
                        score_val = float(match.group(2))
                        job["scores"][f"task_{task_num}"] = score_val

        await asyncio.gather(
            read_stream(process.stdout, False),
            read_stream(process.stderr, True)
        )
        
        await process.wait()
        job["status"] = "done"
        
    except asyncio.TimeoutError:
        job["status"] = "error"
        job["log"] += "\nTimeout executing baseline inference."
    except Exception as e:
        job["status"] = "error"
        job["log"] += f"\nException: {str(e)}"


@app.post("/baseline", description="Run baseline inference script")
async def run_baseline(
    request: Request,
    background_tasks: BackgroundTasks, 
    sync: bool = Query(False, description="Run synchronously and wait for completion")
):
    client_ip = request.client.host if request.client else "unknown"
    allowed, retry_after = await baseline_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum 2 baseline runs per hour. Retry after {retry_after} seconds.",
                "retry_after": retry_after
            }
        )
        
    job_id = str(uuid.uuid4())
    baseline_jobs[job_id] = {
        "status": "running", 
        "scores": {}, 
        "log": "", 
        "started_at": datetime.now(timezone.utc)
    }
    
    if sync:
        task = asyncio.create_task(_run_baseline_job(job_id))
        try:
            await asyncio.wait_for(task, timeout=120.0)
        except asyncio.TimeoutError:
            baseline_jobs[job_id]["status"] = "error"
            baseline_jobs[job_id]["log"] += "\nSync execution timed out after 120s"
        return baseline_jobs[job_id]
        
    background_tasks.add_task(_run_baseline_job, job_id)
    return {
        "job_id": job_id,
        "status": "running",
        "poll_url": f"/baseline/{job_id}"
    }

@app.get("/baseline/{job_id}", description="Get baseline job status")
async def get_baseline_job(job_id: str):
    if job_id not in baseline_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = baseline_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "scores": job["scores"],
        "log": job["log"]
    }

@app.get("/replay/{session_id}", description="Replay a completed episode trajectory")
async def get_replay(session_id: str):
    async with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        env = sessions[session_id]
        
    if not env.state:
        raise HTTPException(status_code=400, detail="Episode not active or initialized")
        
    traj_formatted = []
    
    for t_item in env.state.trajectory:
        obs = t_item.get("observation", {})
        rew = t_item.get("reward", {})
        action = t_item.get("action", {})
        
        traj_formatted.append({
            "step": obs.get("current_step"),
            "action": action,
            "action_status": obs.get("last_action_status", "NONE"),
            "query_results_preview": obs.get("query_results", [])[:3],
            "reward": rew.get("step_reward", 0.0),
            "reward_breakdown": rew.get("reward_breakdown", {}),
            "grader_score_after": rew.get("grader_score_after", 0.0)
        })
        
    return {
        "session_id": session_id,
        "task_id": env.state.task_id,
        "seed": env.state.seed,
        "total_steps": env.state.current_step,
        "final_score": env.grader_score(),
        "trajectory": traj_formatted
    }
