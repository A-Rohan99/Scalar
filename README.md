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

# OpenDataOpsEnv: Autonomous DataOps Incident Response

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-1.111.0-green.svg)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-purple.svg)
![HF Spaces](https://img.shields.io/badge/HF_Spaces-Ready-yellow.svg)

## 📌 Problem Statement & Motivation

Broken data pipelines and database corruption are the #1 cause of data team incidents. When an upstream schema migration silently breaks a downstream SQL view, or when raw Personally Identifiable Information (PII) is accidentally exposed without masking, it can cost enterprises millions of dollars in lost productivity and compromised business intelligence. 

**OpenDataOpsEnv** is a rigorous OpenEnv-compliant RL training and evaluation environment designed to simulate real-world DataOps incident response. Unlike traditional toy grid-world environments, this environment dynamically orchestrates realistic failure scenarios (corrupted records, exposed PII, broken pipeline views) on fully operational SQLite databases generated entirely in memory using randomized Faker seeds. Agents must autonomously query the schema, deduce the anomaly, and write surgical SQL logic to resolve the incident.

---

## 🔄 Environment Overview

OpenDataOpsEnv provides strict OpenEnv spec compliance (Pydantic `Observation`, `Action`, and `Reward` models). To prevent LLMs from hardcoding or "memorizing" table names, absolutely zero table names or column structures are static. The entire environment rebuilds dynamically at runtime using deterministic randomized seeds.

---

## 🎯 Task Descriptions & Difficulty

The environment features three programmatic tasks that scale in difficulty. Graders evaluate dynamically and assign continuous partial-credit scores between `0.0` and `1.0`.

### Task 1: Data Cleaning (Easy)
- **Objective**: Identify the dynamically generated table containing injected `NULL` values within its primary key column. Agents must delete these corrupted rows without wiping out any valid, healthy data.
- **Grader**: Validates total legitimate records remain intact while isolating `NULL` removals.

### Task 2: PII Masking (Medium)
- **Objective**: Locate tables containing unmasked Personally Identifiable Information (emails and phone numbers). Mask the emails to enforce the `a***@domain.com` regex format and phones to the `***-***-XXXX` format using strictly in-place SQL `UPDATE` statements. `DROP COLUMN` is penalized.
- **Grader**: Evaluates regex mask ratios against unaltered field counts.

### Task 3: Pipeline Repair (Hard)
- **Objective**: A previously functional SQL `VIEW` aggregating critical data is shattered because underlying raw table columns were mysteriously altered. Agents must query the internal `error_log` table, filter out noise to find the missing column exception, uncover the raw table schemas, drop the corrupted view, and accurately recreate it with the proper joins.
- **Grader**: Matches the agent's restored pipeline output against the baseline expected data structure.

---

## ⚡ Action Space

The environment exclusively accepts strictly typed JSON actions discriminated by `action_type`.

| Action Type | Structure | Description |
|:---:|:---|:---|
| `query` | `{"action_type": "query", "sql": "..."}` | Executes a safe, read-only SQL `SELECT` statement to inspect schemas or data. |
| `ddl` | `{"action_type": "ddl", "sql": "..."}` | Executes mutating SQL logic (`UPDATE`, `DELETE`, `CREATE`, `DROP`). |
| `test` | `{"action_type": "test", "target_table": "..."}` | Quickly counts the total rows residing in a specified table. |
| `submit` | `{"action_type": "submit"}` | Terminates the episode when the agent has completed the repair. |

---

## 👁️ Observation Space

At every timestep, the agent receives a rich state `Observation`:

| Field | Type | Description |
|:---|:---|:---|
| `current_step` / `max_steps` | Integer | The current interaction step and the hard truncate boundary. |
| `task_id` / `task_description` | Integer / String | The active scenario ID and natural language instructions. |
| `last_action_status` | String | Execution bounds (`SUCCESS`, `ERROR`, `NONE`). |
| `last_error_message` | String | SQLite or Python stack trace message to guide debugging. |
| `query_results` | List[Dict] | A JSON array containing rows returned from the last `query`. |
| `schema_info` | Dict | Real-time map of all active tables and their `CREATE` origin strings. |

*Note: Environments supply dense reward signals (+ for exploration & partial metrics, - for loop repeats & syntax errors).*

---

## 📊 Evaluation Baseline Scores

Inference was executed natively using the OpenAI client architecture on standard models at `temperature=0.0`.

| Task Name | Model | Grader Score |
|:---|:---|:---|
| Data Cleaning (Task 1) | `llama-3.3-70b-versatile` | `1.000` |
| PII Masking (Task 2) | `llama-3.3-70b-versatile` | `0.613` |
| Pipeline Repair (Task 3)| `llama-3.3-70b-versatile` | `0.925` |

---

## 🚀 Setup & Launch Instructions

### Requirements
- Python 3.11+
- Hugging Face / Docker deployment compatibility

### Option A: Local Development (Pip)
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Uvicorn environment:
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 7860
   ```

### Option B: Docker / Hugging Face Spaces Deployment
This application is strictly configured to map to port `7860` natively matching Hugging Face Spaces architectures.
1. Build the lightweight image:
   ```bash
   docker build -t open-dataops-env .
   ```
2. Run the environment detached:
   ```bash
   docker run -d -p 7860:7860 open-dataops-env
   ```

### Running the Baseline
Ensure your environment API is available (default `http://localhost:7860`) and export your `OPENAI_API_KEY`:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
python inference.py
```
