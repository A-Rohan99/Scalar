import os
import re
import json
import sys
import httpx
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
API_BASE_URL  = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME    = os.getenv("MODEL_NAME",   "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")            # no default — required at runtime
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")        # optional: only when using from_docker_image()

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please export your API token before running inference."
    )

# ── OpenAI-compatible client configured via the above variables ──────────────
from openai import OpenAI

# The client automatically picks up OPENAI_API_KEY and OPENAI_BASE_URL from the environment.
client = OpenAI()

# ── System prompt ─────────────────────────────────────────────────────────────
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


# ── Structured stdout logging (START / STEP / END) ────────────────────────────
def log_start(task_id: int, seed: int):
    print(f"START task_id={task_id} seed={seed} model={MODEL_NAME} api_base={API_BASE_URL}")

def log_step(task_id: int, step: int, action_type: str, reward: float, score: float):
    print(f"STEP  task_id={task_id} step={step} action={action_type} reward={reward:.4f} score={score:.4f}")

def log_end(task_id: int, steps: int, final_score: float):
    print(f"END   task_id={task_id} steps={steps} score={final_score:.4f}")
    print(f"SCORE task_{task_id}: {final_score:.4f}")


# ── LLM call ─────────────────────────────────────────────────────────────────
def call_llm(messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"STEP  llm_error={e}")
        sys.exit(1)


# ── Action parsing ────────────────────────────────────────────────────────────
def parse_action(raw_text: str) -> dict:
    """Extract and parse action JSON from LLM output."""
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
            text_fixed = re.sub(r"'([^']*)'", r'"\1"', text)
            return json.loads(text_fixed)
        except json.JSONDecodeError:
            return None


def safe_action(parsed: dict | None, step_num: int) -> dict:
    """Convert parsed dict to a valid environment action."""
    if parsed is None:
        return {"action_type": "submit"}

    action_type = parsed.get("action_type", "").lower()

    if action_type == "query" and "sql" in parsed:
        return parsed
    elif action_type == "ddl" and "sql" in parsed:
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
    seed = BASELINE_SEEDS.get(task_id)
    log_start(task_id, seed)

    try:
        resp = httpx.post(
            f"{API_BASE_URL}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30.0,
        )
        resp.raise_for_status()
        resp_data = resp.json()
        obs = resp_data.get("observation", resp_data)
        session_id = resp_data.get("session_id", "")
    except Exception as e:
        print(f"END   task_id={task_id} steps=0 score=0.0000 error={e}")
        return 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    max_steps = obs.get("max_steps", 25)
    consecutive_parse_failures = 0
    step = 0

    for step in range(max_steps):
        messages.append({"role": "user", "content": json.dumps(obs)})

        try:
            llm_response = call_llm(messages)
            parsed = parse_action(llm_response)

            if parsed is None:
                consecutive_parse_failures += 1
                action = {"action_type": "submit"} if consecutive_parse_failures >= 3 else safe_action(parsed, step)
            else:
                consecutive_parse_failures = 0
                action = safe_action(parsed, step)
        except Exception as e:
            print(f"STEP  task_id={task_id} step={step} llm_error={e}")
            action = {"action_type": "submit"}

        messages.append({"role": "assistant", "content": json.dumps(action)})

        try:
            headers = {"X-Session-ID": session_id} if session_id else {}
            step_resp = httpx.post(
                f"{API_BASE_URL}/step",
                json=action,
                headers=headers,
                timeout=30.0,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            obs = step_data.get("observation", step_data)
            reward = step_data.get("reward", 0.0)
            score  = step_data.get("info", {}).get("grader_score", 0.0)

            log_step(task_id, step + 1, action.get("action_type", "?"), reward, score)

            if step_data.get("done") or step_data.get("truncated"):
                break
        except Exception as e:
            print(f"STEP  task_id={task_id} step={step} env_error={e}")
            break

    # ── Final grader score ────────────────────────────────────────────────────
    try:
        headers = {"X-Session-ID": session_id} if session_id else {}
        grader_resp = httpx.get(f"{API_BASE_URL}/grader", headers=headers, timeout=10.0)
        grader_resp.raise_for_status()
        final_score = grader_resp.json().get("score", 0.0)
    except Exception as e:
        print(f"STEP  task_id={task_id} grader_error={e}")
        final_score = 0.0

    log_end(task_id, step + 1, final_score)
    return final_score


# ── Entry point ───────────────────────────────────────────────────────────────
def run_baseline():
    scores = {}
    for task_id in [1, 2, 3]:
        score = run_task(task_id)
        scores[f"task_{task_id}"] = score

    print("\n--- Summary ---")
    for task, score in scores.items():
        print(f"{task}: {score:.4f}")


if __name__ == "__main__":
    try:
        run_baseline()
    except Exception as e:
        print(f"END   error={e}")
        sys.exit(1)
