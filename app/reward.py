import re

class RewardEngine:
    def __init__(self, task_config: dict):
        self.task_config = task_config
        self.cumulative = 0.0
        self.loop_detector = {}  # hash(sql) -> count
        self.step_history = []
        self.syntax_errors_applied = 0
        self.ssn_found_given = False
        
        # Curiosity signals
        self.queried_tables: set = set()      # tables the agent has queried
        self.queried_columns: set = set()     # columns the agent has used in WHERE/SELECT
        self.query_result_hashes: set = set() # hash of result sets seen

    def compute(self, action: dict, action_result: dict, state_before: dict, state_after: dict, grader_score_before: float, grader_score_after: float) -> tuple[float, dict]:
        breakdown = {}
        sql = action.get("sql", "").strip().lower()
        action_type = action.get("action_type", "")
        task_id = self.task_config.get("task_id")
        
        # 1. curiosity_new_table
        if sql:
            tables_found = [t[0] or t[1] for t in re.findall(r'from\s+(\w+)|join\s+(\w+)', sql)]
            # only reward actual tables that exist in the environment
            valid_tables = state_before.get("tables", {}).keys()
            for t in tables_found:
                if t in valid_tables and t not in self.queried_tables:
                    if len(self.queried_tables) < 3:
                        breakdown["curiosity_new_table"] = round(breakdown.get("curiosity_new_table", 0.0) + 0.08, 2)
                        self.queried_tables.add(t)

        # 2. curiosity_new_result
        if action_type == "query":
            query_results = action_result.get("rows", [])
            if query_results:
                result_hash = hash(str(sorted([str(r) for r in query_results])))
                if result_hash not in self.query_result_hashes and len(self.query_result_hashes) < 5:
                    breakdown["curiosity_new_result"] = 0.03
                    self.query_result_hashes.add(result_hash)

        # 2. null_filter_found
        if task_id == 1 and action_type == "query":
            rows = action_result.get("query_results", action_result.get("rows", []))
            has_null = False
            for row in rows:
                if isinstance(row, dict):
                    if any(val is None for val in row.values()):
                        has_null = True
                        break
            if has_null:
                breakdown["null_filter_found"] = 0.10

        # 3. True grader progress metric
        delta = grader_score_after - grader_score_before
        if delta > 0:
            breakdown["progress"] = round(min(0.5, delta * 2.0), 2)
        elif delta < -0.05:
            breakdown["regression"] = round(max(-0.3, delta * 1.5), 2)

        # 4. syntax_error
        status = action_result.get("status", action_result.get("last_action_status", "SUCCESS"))
        if status == "ERROR":
            if self.syntax_errors_applied < 5:
                breakdown["syntax_error"] = -0.05
                self.syntax_errors_applied += 1

        # 5. destructive_wrong_table
        if action_type == "ddl" and sql:
            tables_in_scope = state_before.get("tables", {}).keys()
            if tables_in_scope and not any(t in sql for t in tables_in_scope):
                breakdown["destructive_wrong_table"] = -0.20

        # 6. loop_penalty
        if sql:
            sql_hash = hash(sql)
            count = self.loop_detector.get(sql_hash, 0) + 1
            self.loop_detector[sql_hash] = count
            if count >= 2:
                breakdown["loop_penalty"] = -0.10

        # 7. efficiency_penalty
        step = state_after.get("current_step", len(self.step_history) + 1)
        if step > 10:
            breakdown["efficiency_penalty"] = -0.01

        # 8. data_destruction
        if task_id == 1:
            total_rows_before = sum(len(rows) for rows in state_before.get("tables", {}).values())
            total_rows_after = sum(len(rows) for rows in state_after.get("tables", {}).values())
            if total_rows_after < total_rows_before:
                breakdown["data_destruction"] = -0.30

        # 9. drop_column_penalty
        if task_id == 2 and sql:
            if "drop column" in sql or "drop table" in sql:
                breakdown["drop_column_penalty"] = -0.50

        # 10. ssn_found bonus (Task 2)
        if task_id == 2 and action_type == "query" and not self.ssn_found_given:
            ssn_col = self.task_config.get("ssn_col", "")
            if ssn_col and ssn_col.lower() in sql:
                breakdown["ssn_found"] = 0.10
                self.ssn_found_given = True

        # 11. partial_mask_penalty — agent NULLed a PII value instead of masking
        if task_id == 2 and action_type == "ddl" and "null" in sql:
            breakdown["partial_mask_penalty"] = -0.10

        # Sum up
        step_reward = sum(breakdown.values())
        
        # Clamp between -1.0 and 1.0
        step_reward = max(-1.0, min(1.0, step_reward))
        
        self.cumulative += step_reward
        self.step_history.append((action, step_reward, breakdown))
        
        return float(step_reward), breakdown
