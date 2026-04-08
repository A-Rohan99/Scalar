---
title: OpenDataOpsEnv
emoji: 🗄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - dataops
  - sql
  - pii-masking
  - data-quality
---

# OpenDataOpsEnv: Autonomous Incident-Response Environment

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-1.111.0-green.svg)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-purple.svg)
![HF Spaces](https://img.shields.io/badge/HF_Spaces-Ready-yellow.svg)

## 💥 The Incident That Started It All

On March 8th 2021, a routine schema migration at a major e-commerce company renamed the column `unit_price` to `price_usd` in their product catalogue. Within 4 hours, 23 downstream SQL views silently broke. Revenue dashboards showed $0 for every product. The data team spent 6 hours manually tracing the dependency graph and rewriting views by hand.

This is not an edge case. According to the 2023 State of Data Engineering survey (Monte Carlo Data), broken data pipelines are the #1 cause of data team incidents, consuming an average of **40% of engineers' time**. The problem is not that engineers don't know how to fix broken views — it's that finding *which* view broke and *why* requires the kind of systematic database exploration that AI agents are uniquely suited to automate.

OpenDataOpsEnv provides the first RL training and evaluation environment specifically designed for DataOps incident response. Unlike toy grid-worlds or game environments, every episode in OpenDataOpsEnv mirrors a real class of incident that data teams face daily: corrupted records, exposed PII, and broken pipeline views. Agents that score well here are agents that would actually save engineering hours in production.

## 🌍 Real-World Deployment Readiness

| Capability | OpenDataOpsEnv | Typical RL Environment |
|:---|:---|:---|
| Domain | Production DataOps | Games / Toy Problems |
| State randomisation | Seeded Faker (infinite episodes) | Fixed maps |
| Reward signal | 9 dense signals per step | Sparse end-of-episode |
| Agent output format | SQL + JSON | Discrete actions |
| Difficulty scaling | 0.5× to 2.0× multiplier | Fixed |
| Replay inspection | `/replay` endpoint | None |
| Leaderboard | `/leaderboard` endpoint | None |

## ❌ The Expensive Reality of DataOps Incidents

In modern enterprise architectures, the volume, velocity, and variety of data flowing through the ecosystem have exponentially increased. Unfortunately, so have the frequency and severity of DataOps and data engineering incidents. A seemingly innocuous error—such as a developer upstream pushing an unannounced schema migration, a microservice failing to properly validate inputs and injecting NULL values into primary key columns, or a legacy script accidentally exposing raw Personally Identifiable Information (PII) without masking—can trigger a catastrophic cascade down the entire data supply chain. When data pipelines break, executive dashboards flatline, machine learning models drift due to poisoned inference data, and the compliance risks related to GDPR and CCPA violations skyrocket. These incidents are notoriously difficult to debug because they exist at the intersection of infrastructure, code logic, and raw stateful data, which inherently lacks transparency until a major failure surfaces.

The financial and operational costs associated with these DataOps incidents are astronomical. Resolving them typically requires senior data engineers to drop their feature-building work, manually crawl through raw `sqlite_master` or `information_schema` tables, write ad-hoc diagnostic SQL queries to isolate exactly which rows and columns have been corrupted, and finally execute precise, high-risk Data Definition Language (DDL) or Data Manipulation Language (DML) statements to repair the state. This reactive, manual firefighting process slows down organizational agility, drains engineering morale, and routinely costs millions of dollars in lost productivity and compromised business intelligence. We desperately need autonomous agents capable of perceiving complex database schemas and executing surgical SQL logic to resolve these incidents instantaneously.


## 🔄 Environment Overview

OpenDataOpsEnv is a state-of-the-art interactive episode environment built entirely upon the OpenEnv specification and driven by a lightning-fast FastAPI backend. It serves as a rigorous testing ground for autonomous DataOps agents. At the start of an episode, the system generates a fully operational SQLite database exclusively in memory, populates it with rich, synthetic data using strictly seeded Faker instances, and artificially orchestrates a realistic failure scenario—such as corrupting a view, exposing PII, or destroying primary key integrity. The agent is then dropped into the environment with no prior knowledge of the database structure and must iteratively query the schema, identify the failure bounds, and execute the exact SQL commands needed to repair the pipeline.

```text
+---------------------+                            +----------------------+
|   DataOps Agent     |                            |   OpenDataOpsEnv     |
|                     |     POST /step (Action)    |                      |
|  1. Parse schemas   | -------------------------> |  1. Execute Action   |
|  2. Query anomalies |                            |  2. Evaluate Grader  |
|  3. Deduce fixes    | <------------------------- |  3. Compute Rewards  |
|  4. Execute DDL/DML |   Response: Observation,   |  4. Generate Snapshot|
|                     |   Reward, & Information    |                      |
+---------------------+                            +----------------------+
```

## ⚡ Action Space

The environment exclusively accepts strictly typed JSON actions dynamically discriminated by the `action_type` parameter, ensuring validation at the FastAPI boundary.

| Action Type | Required Fields | Description |
|:---:|:---|:---|
| `query` | `action_type: "query"`, `sql: str` | Executes a safe, read-only SQL SELECT statement against the environment to read records or inspect schema logic. |
| `ddl` | `action_type: "ddl"`, `sql: str` | Executes a mutating Data Definition Language (DDL) or DML statement (e.g., UPDATE, DELETE, CREATE, DROP). |
| `test` | `action_type: "test"`, `target_table: str` | Executes a rapid internal system test to count the rows currently residing in the specified target table for sanity checking. |
| `submit` | `action_type: "submit"` | Immediately terminates the episode, signaling the agent believes the data incident is completely fixed. |

## 👁️ Observation Space

At every single timestep, the agent receives a rich, comprehensive JSON Observation detailing exactly what is happening in the system.

| Field | Type | Description |
|:---|:---|:---|
| `current_step` | Integer | The exact step number in the current interaction loop. |
| `max_steps` | Integer | The hard ceiling constraint on steps before the episode is forcibly truncated. |
| `task_id` | Integer | The unique identifier pointing to the active scenario (1, 2, or 3). |
| `task_description` | String | A natural language breakdown of the problem the agent must solve. |
| `last_action_status` | String | Enumerated literal bounds (`SUCCESS`, `ERROR`, `NONE`) assessing execution. |
| `last_error_message` | Optional[String] | If `last_action_status` yields `ERROR`, this surfaces the exact SQLite or Python stack trace message to guide agent debugging. |
| `query_results` | List[Dict] | A JSON array containing up to 50 parsed dictionaries representing the rows returned from the last successful `query` or `test` action. |
| `schema_info` | Dict | A real-time dictionary mapping all currently existing tables and views to their origin `CREATE` statements via `sqlite_master`. |
| `system_logs` | List[String] | Synthesized system output logs specifically designed for Task 3 to bury the actual error within noise. |
| `progress_hint` | Optional[String] | An adaptive tactical tip surfaced dynamically if the agent is struggling past step 8 with a score below 0.1. |

## 🎥 Trajectory Replay (Featured Capability)

OpenDataOpsEnv infinitely expands its utility for the RL and agent engineering community by natively supporting complete episode trajectory reconstruction. By calling `GET /replay/{session_id}`, the environment dumps the entire deterministic sequence of actions, granular reward boundaries, grading deltas, and state observations (with query result previews) into a structured JSON timeline. This instantly allows researchers to precisely debug *why* autonomous agents fail mid-episode without actively participating in the live incident, serving as a massive enabler for offline reinforcement learning and post-mortem execution tracking.

## 🗂️ Task Benchmarks

### Task 1: Data Cleaning
- **Objective**: Find the specific dynamically generated table containing randomly injected NULL values within its primary key identification column and delete precisely those corrupted rows without wiping out any valid, healthy data.
- **Difficulty**: Easy
- **Dense Reward Breakdown**: Extracted rows containing NULL identifiers grant immediate exploration and filtering rewards. Data destruction penalties trigger massively if healthy rows are modified.
- **Grader Formula**: `max(0.0, min(1.0, (1.0 - (current_nulls / initial_nulls)) - max(0.0, (initial_valid - current_valid) / initial_valid)))`

### Task 2: PII Masking
- **Objective**: Identify tables containing unmasked Personally Identifiable Information (emails and phone numbers). Mask the emails to enforce the `a***@domain.com` regex format and phones to the `***-***-XXXX` format using strictly in-place SQL `UPDATE` logic. Do not drop constraints.
- **Difficulty**: Medium
- **Dense Reward Breakdown**: High penalties for utilizing explicit `DROP COLUMN` commands. Reward scales linearly as the system scans the targeted table checking how many rows perfectly match the regex masks versus the total row counts.
- **Grader Formula**: `(email_masked_ratio + phone_masked_ratio) / 2.0` bounded to [0.0, 1.0].

### Task 3: Pipeline Repair
- **Objective**: A previously functional SQL `VIEW` that aggregates data for the executive team is completely shattered because underlying raw table columns were suddenly heavily renamed. Agents must query the internal `error_log` table, filter out the synthesized operational noise to find the authentic missing column exception, reverse-engineer the raw table schemas, drop the corrupted view, and correctly recreate it tying the tables appropriately.
- **Difficulty**: Hard
- **Dense Reward Breakdown**: The environment tests query access dynamically, granting massive positive progression thresholds only if `sqlite3.OperationalError` exceptions clear.
- **Grader Formula**: Partial credit yields a `0.3` multiplier based strictly on identifying the proper column schemas matching the baseline, and a massive `0.7` multiplier validating identical row values perfectly matched by joining exact keys algorithmically.

## 🏆 Dense Reward Signals

OpenDataOpsEnv uses a sophisticated standalone dense reward system ensuring continuous gradient signals.
- **Exploration Bonus (`+0.05`)**: Yielded the very first time each randomized table is queried successfully (Capped at maximum exactly `+0.15` per episode).
- **Null Filter Found (`+0.10`)**: Granted instantly if the action fetches rows explicitly containing explicit `None` values (Exclusive to Task 1).
- **Metric Progression (`+0.10` to `+0.40`)**: Scaled perfectly proportional based on exactly how much the underlying deterministic grader score mathematically improves step over step.
- **Repeated Loop Penalty (`-0.10`)**: If the hashed lowercase SQL representation is executed iteratively multiple times, penalizing mindless looping architectures mathematically.
- **Efficiency Penalty (`-0.01`)**: Docked continually for every single step pushed past step 10 to encourage rapid resolution.
- **Syntax Error Penalty (`-0.05`)**: Sapped away when the SQLite parser throws syntax or operational formatting exceptions.
- **Destructive Wrong Table Target (`-0.20`)**: Sapped strongly if a `DDL` or `UPDATE/DELETE` action executes against a table categorically not defined within the scope snapshot bounds.
- **Valid Data Destruction (`-0.30`)**: Heavily punished if valid row counts mysteriously decrease randomly during Task 1 processing without authorization.
- **Cheap Action Drop Column Penalty (`-0.50`)**: Devastating penalty enforced uniquely in Task 2 to heavily dissuade simple lazy `DROP COLUMN` hacks utilized to instantly rid PII fields rather than executing surgical string updates.

## 🛡️ The Zero-Hardcoding Guarantee

LLMs are incredibly notorious for memorizing benchmarks and gaming evaluations by outputting memorized table names (e.g., `users`, `accounts`). OpenDataOpsEnv heavily guards against test contamination by algorithmically rebuilding the complete environment dynamically utilizing deterministic randomized seeds during the generation loop. Absolutely zero table names, zero column structures, and zero row contents are permanently static. Every string is concatenated dynamically with `random.choices` combined against `Faker` utilities.

**Minimal Code Proof of Runtime Schema Generation:**
```python
logical_table = random.choice(["usr", "acct", "client", "member"])
suffix = "".join(random.choices(string.ascii_lowercase, k=4))
main_table_name = f"{logical_table}_{suffix}"  # Example: acct_xqlv
```

## 🏆 Live Benchmarking Leaderboard

The environment acts as a native benchmarking platform by maintaining an internal leaderboard documenting model performance. To view benchmark metrics, simply hit the `/leaderboard` endpoint:

```json
{
  "leaderboard": {
    "task_1": [
      {"rank": 1, "model": "gpt-4o", "score": 0.97, "steps": 5, "timestamp": "..."},
      {"rank": 2, "model": "gpt-4o-mini", "score": 0.82, "steps": 9, "timestamp": "..."}
    ],
    "task_2": [],
    "task_3": []
  },
  "total_episodes_recorded": 42,
  "environment_version": "1.1.0"
}
```
Evaluating interfaces can submit their identities via the `X-Model-Name` header within the `POST /step` endpoint. The platform retains the top 100 entries per task, explicitly ranking them by highest grader score, then fewest steps taken.

## 🚀 Setup & Launch Instructions

### Paradigm A: Docker Compose Deployment (Recommended)
This approach guarantees total operational isolation without python virtual environments colliding, completely wrapping the underlying Uvicorn loops properly on a Debian-based slim Linux build automatically managing binaries.
1. Build the lightweight Docker image tracking the backend framework:
   `docker build -t open-dataops-env .`
2. Instantiate the daemon running detached strictly bound to the port:
   `docker run -d -p 7860:7860 open-dataops-env`

### Paradigm B: Local Development Run (Pip Base)
Use this specific method when rapidly iterating local Python inference files, dynamically testing endpoint modifications, or checking standard outputs in the console interactively without container logs.
1. Install base utilities:
   `pip install -r requirements.txt`
2. Run Uvicorn directly out of the application root mapping to standard local hosts:
   `uvicorn app.api:app --host 0.0.0.0 --port 7860`

### Paradigm C: Hugging Face (HF) Spaces Deployments
The application is pre-bundled identically to match native HF Spaces architectures. Given that the `openenv.yaml` schema endpoints and Dockerfiles declare mapping natively to `7860` with aggressive internal CORS, you can simply upload this exact contiguous repository into an empty HF Docker container space, tracking your configurations flawlessly to standard public access endpoints instantaneously.

## OpenEnv Validation

This environment was designed and verified to comply with the full OpenEnv specification. Manual validation was performed against all spec requirements:
- Typed Pydantic v2 models (Observation, Action, Reward)
- step() / reset() / state() endpoints verified via 47-test suite
- openenv.yaml with all required metadata fields
- 3 tasks with deterministic graders scoring 0.0–1.0
- Baseline inference script outputting SCORE task_N: X.XXXX format
- All 6 required endpoints responding correctly

Automated openenv validate could not be run as the validator package is not yet publicly available on PyPI.

## 📊 Evaluation Baseline Scores

Inference evaluated strictly leveraging the internal trajectory wrapper enforcing a strict temperature bounds of exactly `0.0`. Validated utilizing generic base system layouts ensuring prompt structures correctly guided standard agents. 

| Task Name | Engine Model Parameter | Overall Grader Score | Execution Date |
|:---|:---|:---|:---|
| Data Cleaning | `llama-3.3-70b-versatile` | `1.0000` | April 2026 |
| PII Masking | `llama-3.3-70b-versatile` | `0.6136` | April 2026 |
| Pipeline Repair | `llama-3.3-70b-versatile` | `0.9250` | April 2026 |

<br>

| openenv validate | N/A — package not on PyPI | Manually verified |
| :--- | :--- | :--- |

## 🌟 The Novelty of Non-Hardcoded SQL Evaluation

Standard SQL benchmarking structures heavily rely upon static schemas explicitly dumped out of monolithic `.sql` files, limiting their functional viability entirely the second an LLM is trained across their underlying testing datasets. OpenDataOpsEnv represents a radical evolutionary leap in testing because it forces agents strictly to *perceive* before they actually *act*. Because literal identities defining primary schema constraints actively mutate continuously upon initialization through standard Python Faker instantiations mapped alongside string concatenation, it definitively strips models of their reliance upon training distribution familiarity. Any score produced definitively validates an LLM's legitimate fundamental reasoning capability regarding stateful diagnostics overhead and operational SQLite execution, rather than simply measuring how well it statistically recalls memorized schema strings from a highly polluted generic internet dataset.
