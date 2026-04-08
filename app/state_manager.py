import sqlite3
import random
import uuid
import string
from dataclasses import dataclass, field
from faker import Faker

@dataclass
class EpisodeState:
    db: sqlite3.Connection
    task_id: int
    seed: int
    episode_id: str
    table_registry: dict
    column_registry: dict
    initial_snapshot: dict = field(default_factory=dict)
    current_step: int = 0
    max_steps: int = 20
    done: bool = False
    trajectory: list = field(default_factory=list)
    cumulative_reward: float = 0.0
    difficulty_multiplier: float = 1.0

def generate_episode(task_id: int, seed: int = None, difficulty_multiplier: float = 1.0) -> EpisodeState:
    if seed is None:
        seed = random.randint(0, 999999)
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    db = sqlite3.connect(':memory:')
    db.row_factory = sqlite3.Row
    episode_id = str(uuid.uuid4())

    table_base_pool = ["usr", "acct", "client", "member", "profile"]
    logical_table_name = random.choice(table_base_pool)
    random_suffix = "".join(random.choices(string.ascii_lowercase, k=4))
    main_table_name = f"{logical_table_name}_{random_suffix}"

    col_id = random.choice(["id", "uid", "user_id", "pk"]) + "_" + "".join(random.choices(string.ascii_lowercase, k=2))
    col_name = random.choice(["name", "full_name", "first_last"]) + "_" + "".join(random.choices(string.ascii_lowercase, k=2))
    col_email = random.choice(["email", "mail", "contact_email"]) + "_" + "".join(random.choices(string.ascii_lowercase, k=2))
    col_phone = random.choice(["phone", "phone_number", "mobile"]) + "_" + "".join(random.choices(string.ascii_lowercase, k=2))
    col_created_at = random.choice(["created_at", "inserted_at", "signup_date"]) + "_" + "".join(random.choices(string.ascii_lowercase, k=2))

    table_registry = {"main": main_table_name}
    column_registry = {
        "id": col_id,
        "name": col_name,
        "email": col_email,
        "phone": col_phone,
        "created_at": col_created_at
    }

    cursor = db.cursor()
    correct_df = None

    if task_id == 1:
        cursor.execute(f'''
            CREATE TABLE {main_table_name} (
                {col_id} INTEGER,
                {col_name} TEXT,
                {col_email} TEXT,
                {col_created_at} TEXT
            )
        ''')
        num_rows = random.randint(45, 55)
        num_nulls = random.randint(8, 12)
        
        if difficulty_multiplier <= 0.5:
            num_nulls = random.randint(3, 4)
        elif difficulty_multiplier >= 2.0:
            num_nulls = random.randint(20, 25)
            
        ids = list(range(1, num_rows + 1))
        null_indices = random.sample(range(num_rows), num_nulls)
        for idx in null_indices:
            ids[idx] = None
            
        for i in range(num_rows):
            cursor.execute(
                f"INSERT INTO {main_table_name} ({col_id}, {col_name}, {col_email}, {col_created_at}) VALUES (?, ?, ?, ?)",
                (ids[i], fake.name(), fake.email(), fake.date_time_this_decade().isoformat())
            )

    elif task_id == 2:
        col_ssn = random.choice(["ssn", "tax_id", "national_id", "gov_id"]) + "_" + "".join(random.choices(string.ascii_lowercase, k=2))
        column_registry["ssn_col"] = col_ssn

        cursor.execute(f'''
            CREATE TABLE {main_table_name} (
                {col_id} INTEGER PRIMARY KEY,
                {col_email} TEXT,
                {col_phone} TEXT,
                {col_ssn} TEXT,
                {col_created_at} TEXT
            )
        ''')
        num_rows = random.randint(35, 45)
        if difficulty_multiplier <= 0.5:
            num_rows = 10
        elif difficulty_multiplier >= 2.0:
            num_rows = 200
            
        for i in range(1, num_rows + 1):
            cursor.execute(
                f"INSERT INTO {main_table_name} ({col_id}, {col_email}, {col_phone}, {col_ssn}, {col_created_at}) VALUES (?, ?, ?, ?, ?)",
                (i, fake.email(), fake.phone_number(), fake.ssn(), fake.date_time_this_decade().isoformat())
            )

    elif task_id == 3:
        table_a = f"src_sales_{''.join(random.choices(string.ascii_lowercase, k=4))}"
        table_b = f"src_mapping_{''.join(random.choices(string.ascii_lowercase, k=4))}"
        table_registry["table_a"] = table_a
        table_registry["table_b"] = table_b
        
        old_col1 = "revenue"
        new_col1 = f"rev_{''.join(random.choices(string.ascii_lowercase, k=3))}"
        
        column_registry["old_col_name"] = old_col1
        column_registry["new_col_name"] = new_col1
        
        if difficulty_multiplier >= 2.0:
            old_col2 = "cost"
            new_col2 = f"cst_{''.join(random.choices(string.ascii_lowercase, k=3))}"
            old_col3 = "profit"
            new_col3 = f"prf_{''.join(random.choices(string.ascii_lowercase, k=3))}"
            cursor.execute(f"CREATE TABLE {table_a} (id INTEGER PRIMARY KEY, product_name TEXT, {new_col1} REAL, {new_col2} REAL, {new_col3} REAL, region TEXT)")
        else:
            cursor.execute(f"CREATE TABLE {table_a} (id INTEGER PRIMARY KEY, product_name TEXT, {new_col1} REAL, region TEXT)")
            
        cursor.execute(f"CREATE TABLE {table_b} (id INTEGER PRIMARY KEY, category TEXT)")
        
        num_rows = 20
        for i in range(1, num_rows + 1):
            if difficulty_multiplier >= 2.0:
                cursor.execute(
                    f"INSERT INTO {table_a} (id, product_name, {new_col1}, {new_col2}, {new_col3}, region) VALUES (?, ?, ?, ?, ?, ?)", 
                    (i, fake.word(), round(random.uniform(10.0, 500.0), 2), round(random.uniform(1.0, 100.0), 2), round(random.uniform(1.0, 100.0), 2), fake.state())
                )
            else:
                cursor.execute(
                    f"INSERT INTO {table_a} (id, product_name, {new_col1}, region) VALUES (?, ?, ?, ?)", 
                    (i, fake.word(), round(random.uniform(10.0, 500.0), 2), fake.state())
                )
            cursor.execute(
                f"INSERT INTO {table_b} (id, category) VALUES (?, ?)", 
                (i, random.choice(["A", "B", "C"]))
            )
            
        if difficulty_multiplier >= 2.0:
            correct_df = cursor.execute(
                f"SELECT a.id, a.product_name, a.{new_col1} AS revenue, a.{new_col2} AS cost, a.{new_col3} AS profit, b.category "
                f"FROM {table_a} a JOIN {table_b} b ON a.id = b.id ORDER BY a.id"
            ).fetchall()
        else:
            correct_df = cursor.execute(
                f"SELECT a.id, a.product_name, a.{new_col1} AS revenue, b.category "
                f"FROM {table_a} a JOIN {table_b} b ON a.id = b.id ORDER BY a.id"
            ).fetchall()
            
        view_name = "executive_dashboard"
        table_registry["view"] = view_name
        table_registry["view_name"] = view_name
        
        if difficulty_multiplier >= 2.0:
            cursor.execute(f'''
                CREATE VIEW {view_name} AS 
                SELECT a.id, a.product_name, a.{old_col1}, a.{old_col2}, a.{old_col3}, b.category
                FROM {table_a} a JOIN {table_b} b ON a.id = b.id
            ''')
            cursor.execute(f'''
                CREATE VIEW distractor_view AS 
                SELECT a.id, a.region, a.{old_col1}, b.category
                FROM {table_a} a JOIN {table_b} b ON a.id = b.id
            ''')
        else:
            cursor.execute(f'''
                CREATE VIEW {view_name} AS 
                SELECT a.id, a.product_name, a.{old_col1}, b.category
                FROM {table_a} a JOIN {table_b} b ON a.id = b.id
            ''')
            
        err_table = f"error_log_{''.join(random.choices(string.ascii_lowercase, k=3))}"
        table_registry["error_log"] = err_table
        cursor.execute(f"CREATE TABLE {err_table} (log_id INTEGER PRIMARY KEY, severity TEXT, msg TEXT)")
        
        errors = [
            ("WARNING", "Memory threshold reached on worker node"),
            ("WARNING", "Timeout connecting to upstream replica"),
            ("WARNING", "Garbage collection cycle took >2s"),
            ("WARNING", "User segment cache refreshed with 12ms latency"),
            ("WARNING", "Connection reset by peer during handshake")
        ]
        real_error = ("ERROR", f"View {view_name} references unknown column '{old_col1}'")
        errors.append(real_error)
        random.shuffle(errors)
        
        for sev, msg in errors:
            cursor.execute(f"INSERT INTO {err_table} (severity, msg) VALUES (?, ?)", (sev, msg))

    db.commit()

    state = EpisodeState(
        db=db,
        task_id=task_id,
        seed=seed,
        episode_id=episode_id,
        table_registry=table_registry,
        column_registry=column_registry,
        difficulty_multiplier=difficulty_multiplier,
        initial_snapshot={}
    )
    
    state.initial_snapshot = take_snapshot(state)
    
    if task_id == 3:
        if difficulty_multiplier >= 2.0:
            state.initial_snapshot["expected_view_columns"] = ["id", "product_name", "revenue", "cost", "profit", "category"]
        else:
            state.initial_snapshot["expected_view_columns"] = ["id", "product_name", "revenue", "category"]
            
        state.initial_snapshot["expected_view_data"] = [dict(row) for row in correct_df]

    return state

def get_schema_info(state: EpisodeState) -> dict:
    cursor = state.db.cursor()
    cursor.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')")
    rows = cursor.fetchall()
    
    schema = {}
    for row in rows:
        table = row[0]
        try:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            schema[table] = {
                "columns": [{"name": col['name'], "type": col['type']} for col in columns]
            }
        except sqlite3.OperationalError:
            schema[table] = {"columns": ["[BROKEN_VIEW] Compilation failed"]}
    return schema

def take_snapshot(state: EpisodeState) -> dict:
    cursor = state.db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    tables = [row['name'] for row in cursor.fetchall()]
    
    snapshot = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 100")
            snapshot[table] = [dict(r) for r in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Handle reading from broken views
            snapshot[table] = []
    
    return snapshot
