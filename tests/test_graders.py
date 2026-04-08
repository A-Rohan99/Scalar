import pytest
from app.state_manager import generate_episode
from app.graders import grade_task1, grade_task2, grade_task3
from app.env import DataOpsEnv

def test_task1_initial_score_zero():
    state = generate_episode(1, seed=42)
    score = grade_task1(state.db, state)
    assert score == 0.0

def test_task1_perfect_score():
    state = generate_episode(1, seed=42)
    main_table = state.table_registry["main"]
    id_col = state.column_registry["id"]
    cursor = state.db.cursor()
    cursor.execute(f"DELETE FROM {main_table} WHERE {id_col} IS NULL")
    state.db.commit()
    
    score = grade_task1(state.db, state)
    assert score == 1.0

def test_task1_destruction_penalty():
    state = generate_episode(1, seed=42)
    main_table = state.table_registry["main"]
    cursor = state.db.cursor()
    cursor.execute(f"DELETE FROM {main_table}")
    state.db.commit()
    
    score = grade_task1(state.db, state)
    assert score == 0.0

def test_task2_score_range():
    state = generate_episode(2, seed=42)
    score = grade_task2(state.db, state)
    assert 0.0 <= score <= 1.0

def test_task3_broken_view_score_zero():
    state = generate_episode(3, seed=42)
    score = grade_task3(state.db, state)
    assert score == 0.0

def test_grader_deterministic():
    state = generate_episode(1, seed=42)
    s1 = grade_task1(state.db, state)
    s2 = grade_task1(state.db, state)
    s3 = grade_task1(state.db, state)
    assert s1 == s2 == s3
