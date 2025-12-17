import sqlite3
from pathlib import Path
from src.utils import get_base_path

DB_PATH = get_base_path() / "emg_data.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    # Bảng Patients
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE,
            full_name TEXT,
            gender TEXT,
            birth_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Bảng Recordings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            visit_date TEXT,
            test_name TEXT,
            file_path TEXT,
            duration_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    # Bảng Labels
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER,
            start_ms REAL,
            end_ms REAL,
            trace_id TEXT,
            label_type TEXT,
            p2p_uv REAL,
            rms_uv REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(recording_id) REFERENCES recordings(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS label_definitions (
            code TEXT PRIMARY KEY, -- Ví dụ: PATH, NORM
            name TEXT,             -- Ví dụ: Bệnh lý thần kinh
            color TEXT             -- Ví dụ: #ef4444 (Màu đỏ)
        )
    ''')

    cursor.execute("SELECT count(*) FROM label_definitions")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO label_definitions (code, name, color) VALUES (?, ?, ?)", [
            ("PATH", "Bệnh lý (Pathological)", "#ef4444"),
            ("NORM", "Bình thường (Normal)", "#22c55e"),
            ("ARTIFACT", "Nhiễu (Artifact)", "#eab308")
        ])

    conn.commit()
    conn.close()

def add_patient_if_not_exists(patient_code, full_name, gender=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM patients WHERE patient_code = ?", (patient_code,))
    row = cur.fetchone()
    if row:
        p_id = row['id']
    else:
        cur.execute("INSERT INTO patients (patient_code, full_name, gender) VALUES (?, ?, ?)",
                    (patient_code, full_name, gender))
        p_id = cur.lastrowid
    conn.commit()
    conn.close()
    return p_id

def add_recording(patient_id, visit_date, test_name, file_path, duration_ms=0):
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        SELECT id FROM recordings 
        WHERE patient_id = ? AND visit_date = ? AND test_name = ?
    ''', (patient_id, visit_date, test_name))
    
    row = cur.fetchone()
    
    if row:
        rec_id = row['id']
        cur.execute("UPDATE recordings SET file_path = ? WHERE id = ?", (str(file_path), rec_id))
    else:
        cur.execute('''
            INSERT INTO recordings (patient_id, visit_date, test_name, file_path, duration_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (patient_id, visit_date, test_name, str(file_path), duration_ms))
        rec_id = cur.lastrowid
    
    conn.commit()
    conn.close()
    return rec_id

def get_all_recordings():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT r.id, r.visit_date, r.test_name, p.full_name, p.patient_code
        FROM recordings r
        JOIN patients p ON r.patient_id = p.id
        ORDER BY r.created_at DESC
    ''')
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_recording_by_id(rec_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT r.*, p.full_name, p.patient_code, p.gender 
        FROM recordings r
        JOIN patients p ON r.patient_id = p.id
        WHERE r.id = ?
    ''', (rec_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

# Display Label
def get_labels_by_recording(rec_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM labels WHERE recording_id = ? ORDER BY start_ms ASC", (rec_id,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Save Label
def save_label_to_db(recording_id, start, end, trace, label, p2p, rms):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO labels (recording_id, start_ms, end_ms, trace_id, label_type, p2p_uv, rms_uv)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (recording_id, start, end, trace, label, p2p, rms))
    conn.commit()
    conn.close()

# Delete Label
def delete_label_by_id(label_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM labels WHERE id = ?", (label_id,))
    conn.commit()
    conn.close()

def get_all_label_defs():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM label_definitions")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_label_def(code, name, color="#6b7280"):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO label_definitions (code, name, color) VALUES (?, ?, ?)", (code, name, color))
        conn.commit()
        return True, "Thêm thành công"
    except sqlite3.IntegrityError:
        return False, "Mã nhãn đã tồn tại!"
    finally:
        conn.close()

def delete_all_labels_by_recording(rec_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM labels WHERE recording_id = ?", (rec_id,))
    conn.commit()
    conn.close()