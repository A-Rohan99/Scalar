from app.models import QueryAction, DDLAction, TestAction, SubmitAction

TASK_REGISTRY = {
    1: {
        "id": 1,
        "name": "Data Cleaning",
        "difficulty": "easy",
        "description": "Find the table containing NULL values in its ID column and remove only those rows, without deleting any valid data.",
        "max_steps": 15,
        "success_threshold": 0.9,
        "hints": [
            "Start by querying sqlite_master to see all tables",
            "Use SELECT * FROM table WHERE col IS NULL to find the problem",
            "Use DELETE FROM table WHERE col IS NULL to fix it"
        ]
    },
    2: {
        "id": 2, 
        "name": "PII Masking",
        "difficulty": "medium",
        "description": "Find tables containing email addresses, phone numbers, and government ID numbers (SSN/tax_id). Mask each field using SQL string functions preserving the original value length. Emails → a***@domain.com, phones → ***-***-XXXX, SSN/gov IDs → ***-**-XXXX. Do not drop any columns or NULL any values.",
        "max_steps": 25,
        "success_threshold": 0.80,
        "hints": [
            "Query the schema first — there are three PII columns to find",
            "Use SUBSTR and REPLACE SQL functions for masking, not NULL or DROP",
            "Email mask: first char + asterisks + @domain (same total length)",
            "SSN mask pattern: ***-**-XXXX (keep last 4 digits)"
        ]
    },
    3: {
        "id": 3,
        "name": "Pipeline Repair", 
        "difficulty": "hard",
        "description": "A SQL VIEW used by the executive dashboard is broken. Inspect the error_log table to find the real error among noise, identify the renamed columns in the source tables, and recreate the VIEW correctly.",
        "max_steps": 25,
        "success_threshold": 0.75,
        "hints": [
            "Read the error_log table \u2014 filter for severity='ERROR'",
            "Query sqlite_master to see the broken VIEW definition",
            "Query the raw tables to find the current correct column names",
            "DROP the VIEW then CREATE it with corrected column names"
        ]
    }
}

def get_task(task_id: int) -> dict:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"task_id {task_id} not found. Valid: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]

def get_action_schema() -> dict:
    return {
        "query": QueryAction.model_json_schema(),
        "ddl": DDLAction.model_json_schema(),
        "test": TestAction.model_json_schema(),
        "submit": SubmitAction.model_json_schema()
    }
