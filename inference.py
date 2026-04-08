import os
import re
import json
import sys
import httpx
from typing import List, Optional
from dotenv import load_dotenv

try:
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
except NameError:
    _root = os.getcwd()

if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv()

# ── Required environment variables (as per submission spec) ──────────────────
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# ── Local Environment Backend Mapping ──────────────────────────────────────────
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI-compatible client configured via the above variables ──────────────
from openai import OpenAI

client = None
try:
    if API_BASE_URL and API_KEY:
        # Crucial for Phase 2 constraint: must route exactly through their base proxy
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    elif API_KEY:
        client = OpenAI(api_key=API_KEY)
    else:
        client = OpenAI()
except Exception:
    pass

# ── System prompt ──
try:
    from prompts import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = (
        "You are a DataOps incident-response agent. "
        "Diagnose and repair the database environment by issuing SQL actions. "
        "Reply ONLY with a JSON object: "
        "{\"action_type\": \"query\"|\"ddl\"|\"test\"|\"submit\", \"sql\": \"...\"}. "
        "When the task is complete, issue {\"action_type\": \"submit\"}."
    )

BASELINE_SEEDS = {1: 42, 2: 99, 3: 777}
BENCHMARK_NAME = "opendataopsenv"
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Structured stdout logging ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    error_val = str(error_val).replace('\n', ' ').replace('\r', ' ')
    done_val = str(bool(done)).lower()
    action_clean = str(action).replace('\n', ' ').replace('\r', ' ')
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(messages: list) -> str:
    try:
        if client is None:
            return '{"action_type": "submit"}'
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return '{"action_type": "submit"}'

# ── Action parsing ────────────────────────────────────────────────────────────
def parse_action(raw_text: str) -> dict:
    if raw_text is None:
        return None
    text = raw_text.strip()
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence_match:
        text = fence_match.group(1).strip()
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        text = brace_match.group(0)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r"'([^']*)'", r'"\1"', text))
        except json.JSONDecodeError:
            return None

def safe_action(parsed: dict | None, step_num: int) -> dict:
    if parsed is None:
        return {"action_type": "submit"}
    action_type = parsed.get("action_type", "").lower()
    if action_type in ["query", "ddl"] and "sql" in parsed:
        return parsed
    elif action_type == "test" and "target_table" in parsed:
        return parsed
    elif action_type == "submit":
        return parsed
    elif "sql" in parsed:
        sql = parsed["sql"].strip().upper()
        inferred_type = "query" if sql.startswith(("SELECT", "WITH", "EXPLAIN")) else "ddl"
        return {"action_type": inferred_type, "sql": parsed["sql"]}
    else:
        if step_num <= 3:
            return {"action_type": "query", "sql": "SELECT name, sql FROM sqlite_master WHERE type IN ('table','view')"}
        return {"action_type": "submit"}

# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task_id: int) -> float:
    seed = BASELINE_SEEDS.get(task_id, 42)
    task_name = f"task_{task_id}"
    log_start(task=task_name, env=BENCHMARK_NAME, model=MODEL_NAME)

    rewards_list = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        resp = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30.0)
        resp.raise_for_status()
        resp_data = resp.json()
        obs = resp_data.get("observation", resp_data)
        session_id = resp_data.get("session_id", "")
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_steps = obs.get("max_steps", 25)
    consecutive_failures = 0
    
    for step in range(1, max_steps + 1):
        messages.append({"role": "user", "content": json.dumps(obs)})

        llm_response = call_llm(messages)
        parsed = parse_action(llm_response)
        
        if parsed is None:
            consecutive_failures += 1
            action = {"action_type": "submit"} if consecutive_failures >= 3 else safe_action(parsed, step)
        else:
            consecutive_failures = 0
            action = safe_action(parsed, step)

        messages.append({"role": "assistant", "content": json.dumps(action)})

        try:
            headers = {"X-Session-ID": session_id} if session_id else {}
            step_resp = httpx.post(
                f"{ENV_BASE_URL}/step", json=action, headers=headers, timeout=30.0
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            obs = step_data.get("observation", step_data)
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False) or step_data.get("truncated", False))
            
            error_val = None
            if "last_error_message" in obs and obs["last_error_message"]:
                error_val = obs["last_error_message"]
            
            score = float(step_data.get("info", {}).get("grader_score", 0.0))
            
            rewards_list.append(reward)
            steps_taken = step
            
            log_step(step=step, action=json.dumps(action), reward=reward, done=done, error=error_val)

            if done:
                break
        except Exception as e:
            log_step(step=step, action=json.dumps(action), reward=0.0, done=True, error=str(e))
            rewards_list.append(0.0)
            steps_taken = step
            break

    try:
        if session_id:
            grader_resp = httpx.get(f"{ENV_BASE_URL}/grader", headers={"X-Session-ID": session_id}, timeout=10.0)
            if grader_resp.status_code == 200:
                score = float(grader_resp.json().get("score", score))
    except Exception:
        pass

    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)
    return score

# ── Entry point ───────────────────────────────────────────────────────────────
def run_baseline():
    for task_id in [1, 2, 3]:
        run_task(task_id)

if __name__ == "__main__":
    try:
        run_baseline()
    except Exception:
        pass
