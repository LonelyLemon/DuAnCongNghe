import sqlite3
import json
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

    try:
        cursor.execute("ALTER TABLE recordings ADD COLUMN clinical_conclusion TEXT")
        print("--- [DB] Đã thêm cột 'clinical_conclusion' vào bảng recordings ---")
    except sqlite3.OperationalError:
        pass

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

    try:
        cursor.execute("ALTER TABLE recordings ADD COLUMN metadata TEXT")
        print("--- [DB] Đã thêm cột 'metadata' ---")
    except sqlite3.OperationalError:
        pass

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

def add_recording(patient_id, visit_date, test_name, file_path, metadata=None):
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT id FROM recordings WHERE file_path = ?", (file_path,))
    res = cur.fetchone()
    if res:
        conn.close()
        return res['id']

    meta_json = json.dumps(metadata) if metadata else "{}"

    cur.execute('''
        INSERT INTO recordings (patient_id, visit_date, test_name, file_path, created_at, metadata)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
    ''', (patient_id, visit_date, test_name, file_path, meta_json))
    
    rec_id = cur.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def get_recording_metadata(rec_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT metadata FROM recordings WHERE id = ?", (rec_id,))
    row = cur.fetchone()
    conn.close()
    if row and row['metadata']:
        return json.loads(row['metadata'])
    return {}

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

def delete_recording(rec_id):
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT file_path FROM recordings WHERE id = ?", (rec_id,))
        row = cur.fetchone()
        file_path = row['file_path'] if row else None
        
        cur.execute("DELETE FROM labels WHERE recording_id = ?", (rec_id,))
        
        cur.execute("DELETE FROM recordings WHERE id = ?", (rec_id,))
        
        conn.commit()
        return file_path
    except Exception as e:
        print(f"Lỗi khi xóa DB: {e}")
        return None
    finally:
        conn.close()

def update_recording_conclusion(rec_id, text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE recordings SET clinical_conclusion = ? WHERE id = ?", (text, rec_id))
    conn.commit()
    conn.close()

def update_recording_metadata(rec_id, new_visit_date, new_test_name, new_patient_name, new_patient_code):
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT patient_id FROM recordings WHERE id = ?", (rec_id,))
        row = cur.fetchone()
        if not row: return False, "Không tìm thấy bản ghi"
        
        p_id = row['patient_id']
        
        cur.execute('''
            UPDATE recordings 
            SET visit_date = ?, test_name = ? 
            WHERE id = ?
        ''', (new_visit_date, new_test_name, rec_id))
        
        cur.execute('''
            UPDATE patients 
            SET full_name = ?, patient_code = ? 
            WHERE id = ?
        ''', (new_patient_name, new_patient_code, p_id))
        
        conn.commit()
        return True, "Cập nhật thành công"
    except Exception as e:
        print(f"Lỗi Update DB: {e}")
        return False, str(e)
    finally:
        conn.close()