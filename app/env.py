import random
import asyncio
import re
from datetime import datetime, timezone
from typing import Optional, Tuple
import sqlite3

from app.models import Action, Observation, Reward, StateSnapshot
from app.state_manager import EpisodeState, generate_episode, get_schema_info, take_snapshot
from app.reward import RewardEngine
from app.tasks import get_task
from app.graders import grade_task1, grade_task2, grade_task3

def validate_sql(sql: str, action_type: str) -> Tuple[bool, str]:
    if not sql:
        return False, "Empty SQL statement"
        
    sql_upper = sql.strip().upper()
    tokens = sql_upper.split()
    if not tokens:
        return False, "Empty SQL statement"
        
    first_token = tokens[0]

    if action_type == "query":
        allowed_starts = {"SELECT", "WITH", "EXPLAIN", "PRAGMA"}
        if first_token not in allowed_starts:
            return False, "Only SELECT statements allowed in query actions"
            
        blocked_patterns = [
            r"DROP\s+TABLE", 
            r"DELETE\s+FROM\s+SQLITE_MASTER", 
            r"DROP\s+INDEX", 
            r"ATTACH\s+DATABASE"
        ]
        for pat in blocked_patterns:
            if re.search(pat, sql_upper):
                clean_pat = pat.replace('\\s+', ' ')
                return False, f"Blocked pattern detected in query: {clean_pat}"
                
        return True, ""

    elif action_type == "ddl":
        if re.search(r"\bDROP\s+TABLE\b", sql_upper):
            return False, "DROP TABLE is blocked"
        if re.search(r"\bATTACH\b|\bDETACH\b", sql_upper):
            return False, "ATTACH and DETACH are blocked"
        if re.search(r"(UPDATE|INSERT\s+INTO|DELETE\s+FROM)\s+(SQLITE_MASTER|SQLITE_SEQUENCE)\b", sql_upper):
            return False, "Writing to sqlite_master or sqlite_sequence is blocked"
            
        if first_token == "ALTER":
            if not re.search(r"^ALTER\s+TABLE\s+.*?\s+RENAME\s+COLUMN", sql_upper):
                return False, "Only ALTER TABLE ... RENAME COLUMN is allowed"
                
        if first_token == "CREATE":
            if re.search(r"^CREATE\s+TABLE", sql_upper) and not re.search(r"^CREATE\s+(TEMP|TEMPORARY)\s+TABLE", sql_upper):
                return False, "Only temporary tables are allowed (CREATE TEMP TABLE)"
            if not (re.search(r"^CREATE\s+(TEMP|TEMPORARY)\s+TABLE", sql_upper) or re.search(r"^CREATE\s+VIEW", sql_upper)):
                 return False, "Only CREATE VIEW or CREATE TEMP TABLE allowed for CREATE"
                 
        if first_token == "DROP":
            if not re.search(r"^DROP\s+VIEW", sql_upper):
                return False, "Only DROP VIEW is allowed for DROP statements"
                
        allowed_starts = {"UPDATE", "INSERT", "DELETE", "ALTER", "CREATE", "DROP"}
        if first_token not in allowed_starts:
            return False, f"DDL action does not allow '{first_token}' statements"
                
        return True, ""
        
    return True, ""

class DataOpsEnv:
    def __init__(self):
        self.state: Optional[EpisodeState] = None
        self.reward_engine: Optional[RewardEngine] = None
        self.task_config: Optional[dict] = None
        self._lock = asyncio.Lock()
        self.last_activity = datetime.now(timezone.utc)
        self._last_grader_score = None

    async def reset(self, task_id: int, seed: int = None, difficulty_multiplier: float = 1.0) -> Observation:
        async with self._lock:
            self.last_activity = datetime.now(timezone.utc)
            if task_id not in [1, 2, 3]:
                raise ValueError("task_id must be 1, 2, or 3")
                
            self.state = generate_episode(task_id, seed, difficulty_multiplier)
            task_info = get_task(task_id)
            
            self.task_config = {
                "task_id": task_id,
            }
            
            main_table = self.state.table_registry.get("main")
            if task_id == 1:
                id_col = self.state.column_registry.get("id")
                rows = self.state.initial_snapshot.get(main_table, [])
                self.task_config["initial_null_count"] = sum(1 for r in rows if r.get(id_col) is None)
            elif task_id == 2:
                rows = self.state.initial_snapshot.get(main_table, [])
                self.task_config["total_rows"] = len(rows)
                self.task_config["pii_columns"] = [self.state.column_registry.get("email"), self.state.column_registry.get("phone")]
                self.task_config["ssn_col"] = self.state.column_registry.get("ssn_col")
            elif task_id == 3:
                self.task_config["expected_view_output"] = True
                
            self.reward_engine = RewardEngine(self.task_config)
            
            system_logs = []
            if task_id == 3:
                err_table = self.state.table_registry.get("error_log")
                if err_table:
                    try:
                        cursor = self.state.db.cursor()
                        cursor.execute(f"SELECT msg FROM {err_table}")
                        system_logs = [r["msg"] for r in cursor.fetchall()]
                    except Exception:
                        pass

            self._last_grader_score = self.grader_score()

            return Observation(
                current_step=0,
                max_steps=self.state.max_steps,
                task_id=task_id,
                task_description=task_info["description"],
                last_action_status="NONE",
                last_error_message=None,
                query_results=[],
                results_truncated=False,
                total_rows_returned=0,
                schema_info=get_schema_info(self.state),
                system_logs=system_logs[:20],
                logs_truncated=len(system_logs) > 20,
                progress_hint=None
            )

    def grader_score(self) -> float:
        if not self.state:
            return 0.0
        if self.state.task_id == 1:
            return grade_task1(self.state.db, self.state)
        elif self.state.task_id == 2:
            return grade_task2(self.state.db, self.state)
        elif self.state.task_id == 3:
            return grade_task3(self.state.db, self.state)
        return 0.0

    def get_state(self) -> StateSnapshot:
        if not self.state:
            raise ValueError("Environment not initialized")
        
        tables = take_snapshot(self.state)
        return StateSnapshot(
            episode_id=self.state.episode_id,
            task_id=self.state.task_id,
            current_step=self.state.current_step,
            tables=tables,
            trajectory=self.state.trajectory,
            grader_score=self.grader_score(),
            seed=self.state.seed,
            difficulty_multiplier=self.state.difficulty_multiplier
        )

    async def step(self, action: Action, session_id: str = "") -> Tuple[Observation, Reward]:
        async with self._lock:
            try:
                self.last_activity = datetime.now(timezone.utc)
                if not self.state or self.state.done:
                    raise RuntimeError("Episode is not active. Call reset().")

                score_before = getattr(self, "_last_grader_score", None)
                if score_before is None:
                    score_before = self.grader_score()

                try:
                    action_dict = action.model_dump()
                except AttributeError:
                    action_dict = action if isinstance(action, dict) else dict(action)

                action_type = getattr(action, "action_type", action_dict.get("action_type"))

                state_before = self.get_state().model_dump()

                action_result = {
                    "status": "SUCCESS",
                    "error_message": None,
                    "rows": [],
                    "results_truncated": False,
                    "total_rows_returned": 0
                }

                sql = getattr(action, "sql", action_dict.get("sql", ""))

                is_valid = True
                val_msg = ""
                if action_type in ["query", "ddl"]:
                    is_valid, val_msg = validate_sql(sql, action_type)

                if not is_valid:
                    action_result["status"] = "ERROR"
                    action_result["error_message"] = val_msg
                else:
                    self.state.current_step += 1
                    try:
                        cursor = self.state.db.cursor()
                        if action_type == "query":
                            cursor.execute(sql)
                            all_rows = cursor.fetchall()
                            total = len(all_rows)
                            display_rows = all_rows[:10]  # hard cap at 10

                            def truncate_value(v, max_len=100):
                                if v is None: return None
                                s = str(v)
                                return s[:max_len] + "..." if len(s) > max_len else s

                            col_names = [d[0] for d in cursor.description] if cursor.description else []

                            result_dicts = [
                                {col: truncate_value(val) for col, val in zip(col_names, row)}
                                for row in display_rows
                            ]

                            action_result["rows"] = result_dicts
                            action_result["results_truncated"] = total > 10
                            action_result["total_rows_returned"] = total
                        elif action_type == "ddl":
                            cursor.execute(sql)
                            self.state.db.commit()
                        elif action_type == "test":
                            target_table = getattr(action, "target_table", action_dict.get("target_table"))
                            cursor.execute(f"SELECT COUNT(*) as cnt FROM {target_table}")
                            action_result["rows"] = [dict(r) for r in cursor.fetchall()]
                        elif action_type == "submit":
                            self.state.done = True
                    except Exception as e:
                        action_result["status"] = "ERROR"
                        action_result["error_message"] = str(e)

                score_after = self.grader_score()
                self._last_grader_score = score_after

                state_after = self.get_state().model_dump()
                state_after["grader_score"] = score_after

                step_reward_val, breakdown = self.reward_engine.compute(
                    action=action_dict,
                    action_result=action_result,
                    state_before=state_before,
                    state_after=state_after,
                    grader_score_before=score_before,
                    grader_score_after=score_after
                )

                truncated = False
                if self.state.current_step >= self.state.max_steps:
                    truncated = True
                    self.state.done = True

                progress_hint = None
                if self.state.current_step > 8 and score_after < 0.1:
                    task_info = get_task(self.state.task_id)
                    hints = task_info.get("hints", [])
                    progress_hint = random.choice(hints) if hints else "Review the schema and target carefully."

                system_logs = []
                if self.state.task_id == 3:
                    err_table = self.state.table_registry.get("error_log")
                    if err_table:
                        try:
                            c = self.state.db.cursor()
                            c.execute(f"SELECT msg FROM {err_table}")
                            system_logs = [r["msg"] for r in c.fetchall()]
                        except Exception:
                            pass

                obs = Observation(
                    current_step=self.state.current_step,
                    max_steps=self.state.max_steps,
                    task_id=self.state.task_id,
                    task_description=get_task(self.state.task_id)["description"],
                    last_action_status=action_result["status"],
                    last_error_message=action_result["error_message"],
                    query_results=action_result["rows"],
                    results_truncated=action_result.get("results_truncated", False),
                    total_rows_returned=action_result.get("total_rows_returned", 0),
                    schema_info=get_schema_info(self.state),
                    system_logs=system_logs[:20],
                    logs_truncated=len(system_logs) > 20,
                    progress_hint=progress_hint
                )

                reward = Reward(
                    step_reward=step_reward_val,
                    cumulative_reward=self.reward_engine.cumulative,
                    reward_breakdown=breakdown,
                    done=self.state.done,
                    truncated=truncated,
                    grader_score_before=score_before,
                    grader_score_after=score_after
                )

                self.state.trajectory.append({
                    "action": action_dict,
                    "observation": obs.model_dump(),
                    "reward": reward.model_dump()
                })

                return obs, reward

            except sqlite3.OperationalError as e:
                # SQL syntax errors, missing tables, broken views
                return self._error_observation(
                    error_msg=f"SQL error: {str(e)}",
                    reward_penalty=-0.05
                ), self._error_reward(breakdown={"sql_error": -0.05})
                
            except sqlite3.DatabaseError as e:
                # Corrupted state, PRAGMA failures, trigger issues
                return self._error_observation(
                    error_msg=f"Database error: {str(e)}",
                    reward_penalty=-0.10
                ), self._error_reward(breakdown={"db_error": -0.10})
                
            except Exception as e:
                # Catch-all: unknown agent-triggered edge cases
                # Log the full traceback internally but NEVER expose it
                import traceback
                internal_log = traceback.format_exc()
                # Store in state for debugging but do not return to agent
                if self.state:
                    self.state.trajectory.append({
                        "step": self.state.current_step,
                        "internal_error": internal_log[:500]
                    })
                return self._error_observation(
                    error_msg="Internal error — action could not be processed",
                    reward_penalty=-0.05
                ), self._error_reward(breakdown={"internal_error": -0.05})

    def _error_observation(self, error_msg: str, reward_penalty: float) -> Observation:
        return Observation(
            current_step=self.state.current_step if self.state else 0,
            max_steps=self.state.max_steps if self.state else 20,
            task_id=self.state.task_id if self.state else 0,
            task_description="",
            last_action_status="ERROR",
            last_error_message=error_msg,
            query_results=[],
            schema_info={},
            system_logs=[f"ERROR: {error_msg}"],
            results_truncated=False,
            total_rows_returned=0,
            progress_hint=None
        )

    def _error_reward(self, breakdown: dict) -> Reward:
        step_reward = sum(breakdown.values())
        if self.state:
            self.state.cumulative_reward += step_reward
        return Reward(
            step_reward=step_reward,
            cumulative_reward=self.state.cumulative_reward if self.state else step_reward,
            reward_breakdown=breakdown,
            done=False,
            truncated=False,
            grader_score_before=0.0,
            grader_score_after=0.0
        )
