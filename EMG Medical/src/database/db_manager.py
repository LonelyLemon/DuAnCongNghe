import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent.parent / "emg_data.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Khởi tạo các bảng dữ liệu nếu chưa tồn tại"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_code TEXT UNIQUE, -- Mã bệnh nhân (VD: 2508182938)
            full_name TEXT,
            gender TEXT,
            birth_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            visit_date TEXT,
            test_name TEXT,
            file_path TEXT, -- Đường dẫn file txt gốc
            duration_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER,
            start_ms REAL,
            end_ms REAL,
            trace_id TEXT,
            label_type TEXT, -- Pathological, Normal, Artifact...
            p2p_uv REAL,
            rms_uv REAL,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(recording_id) REFERENCES recordings(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def add_patient_if_not_exists(patient_code, full_name, gender=None):
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT id FROM patients WHERE patient_code = ?", (patient_code,))
    row = cur.fetchone()
    
    if row:
        patient_id = row['id']
    else:
        cur.execute("INSERT INTO patients (patient_code, full_name, gender) VALUES (?, ?, ?)",
                    (patient_code, full_name, gender))
        patient_id = cur.lastrowid
    
    conn.commit()
    conn.close()
    return patient_id

def add_recording(patient_id, visit_date, test_name, file_path, duration_ms=0):
    conn = get_connection()
    cur = conn.cursor()
    
    # Kiểm tra tránh trùng lặp file
    cur.execute("SELECT id FROM recordings WHERE file_path = ?", (str(file_path),))
    row = cur.fetchone()
    
    if row:
        return row['id']
        
    cur.execute('''
        INSERT INTO recordings (patient_id, visit_date, test_name, file_path, duration_ms)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, visit_date, test_name, str(file_path), duration_ms))
    
    rec_id = cur.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def get_all_recordings():
    """Lấy danh sách tất cả bản ghi để hiển thị Dashboard"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT r.id, r.visit_date, r.test_name, p.full_name, p.patient_code, r.file_path
        FROM recordings r
        JOIN patients p ON r.patient_id = p.id
        ORDER BY r.visit_date DESC
    ''')
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_recording_by_id(rec_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def save_label_to_db(recording_id, start, end, trace, label, p2p, rms):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO labels (recording_id, start_ms, end_ms, trace_id, label_type, p2p_uv, rms_uv)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (recording_id, start, end, trace, label, p2p, rms))
    conn.commit()
    conn.close()