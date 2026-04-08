from typing import List, Dict, Optional, Literal, Union, Annotated, Any
from pydantic import BaseModel, Field, ConfigDict

class QueryAction(BaseModel):
    model_config = ConfigDict(strict=True)
    action_type: Literal["query"] = Field(..., description="The type of action, must be 'query'")
    sql: str = Field(..., description="The SQL query to execute")

class DDLAction(BaseModel):
    model_config = ConfigDict(strict=True)
    action_type: Literal["ddl"] = Field(..., description="The type of action, must be 'ddl'")
    sql: str = Field(..., description="The DDL SQL statement to execute")

class TestAction(BaseModel):
    model_config = ConfigDict(strict=True)
    action_type: Literal["test"] = Field(..., description="The type of action, must be 'test'")
    target_table: str = Field(..., description="The target table to run tests against")

class SubmitAction(BaseModel):
    model_config = ConfigDict(strict=True)
    action_type: Literal["submit"] = Field(..., description="The type of action, must be 'submit'")

Action = Annotated[Union[QueryAction, DDLAction, TestAction, SubmitAction], Field(discriminator='action_type', description="Union of all four actions, discriminated by action_type")]

class Observation(BaseModel):
    model_config = ConfigDict(strict=True)
    current_step: int = Field(..., description="The current step in the episode")
    max_steps: int = Field(..., description="The maximum number of steps allowed in the episode")
    task_id: int = Field(..., description="The unique identifier for the current task")
    task_description: str = Field(..., description="The description of the task")
    last_action_status: Literal["SUCCESS", "ERROR", "NONE"] = Field(..., description="The status of the last executed action")
    last_error_message: Optional[str] = Field(None, description="The error message from the last action, if any")
    query_results: List[Dict[str, Any]] = Field(default_factory=list, description="Up to 10 rows from the last query result")
    results_truncated: bool = Field(default=False, description="True if query returned more rows than shown")
    total_rows_returned: int = Field(default=0, description="Actual row count before truncation")
    schema_info: Dict[str, Any] = Field(..., description="Column names and types only — not data")
    system_logs: List[str] = Field(..., max_length=20, description="A list of system logs")
    logs_truncated: bool = Field(default=False, description="True if there were more logs than shown")
    progress_hint: Optional[str] = Field(None, description="A hint for the progress of the task, if available")

class Reward(BaseModel):
    model_config = ConfigDict(strict=True)
    step_reward: float = Field(..., ge=-1.0, le=1.0, description="The reward for the current step, between -1.0 and 1.0")
    cumulative_reward: float = Field(..., description="The total cumulative reward accumulated so far")
    reward_breakdown: Dict[str, float] = Field(..., description="A breakdown of the components contributing to the reward")
    done: bool = Field(..., description="Whether the episode has finished successfully or not")
    truncated: bool = Field(..., description="Whether the episode was truncated (e.g., maximum steps reached)")
    grader_score_before: float = Field(..., description="Grader score before the action")
    grader_score_after: float = Field(..., description="Grader score after the action")

class StateSnapshot(BaseModel):
    model_config = ConfigDict(strict=True)
    episode_id: str = Field(..., description="The unique identifier of the episode")
    task_id: int = Field(..., description="The unique identifier of the task")
    current_step: int = Field(..., description="The current step count")
    tables: Dict[str, List[Dict[str, Any]]] = Field(..., description="The contents of the tables currently in the environment")
    trajectory: List[Dict[str, Any]] = Field(..., description="The trajectory of actions and observations collected so far")
    grader_score: float = Field(..., description="The current score assigned by the grader")
    seed: int = Field(..., description="The random seed used for the current state snapshot")
    difficulty_multiplier: float = Field(1.0, description="Task difficulty curriculum multiplier")

class CompletedEpisode(BaseModel):
    model_config = ConfigDict(strict=True)
    episode_id: str
    task_id: int
    total_steps: int
    final_score: float
    failed_actions: List[str]
