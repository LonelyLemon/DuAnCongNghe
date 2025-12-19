import numpy as np
import pandas as pd
import re
import io
from pathlib import Path

# --- A. XỬ LÝ TEXT / NATUS ---
def validate_natus_structure(content_str: str) -> bool:
    """Kiểm tra chữ ký file Natus"""
    if not content_str: 
        return False
    # FIX: Giảm bớt keyword bắt buộc để linh hoạt hơn (bỏ EMG vì có thể file chỉ có NCS)
    keywords = ["Patient ID", "Sampling Frequency", "Export Filter"]
    for k in keywords:
        if k not in content_str: 
            return False
    return True

def extract_numbers_robust(text_chunk: str):
    # Tìm tất cả các số (bao gồm số âm, số thập phân dùng chấm hoặc phẩy)
    pattern = r'[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?'
    matches = re.findall(pattern, text_chunk)
    clean_nums = []
    for m in matches:
        try:
            # FIX: Chuyển đổi dấu phẩy thành chấm để Python hiểu là số thực
            val = float(m.replace(',', '.'))
            clean_nums.append(val)
        except:
            continue
            
    return np.array(clean_nums)

def parse_natus_content(content_str: str):
    info = {}
    patterns = {
        'patient_id': r'Patient ID\s*=\s*(.*)',
        'first_name': r'First Name\s*=\s*(.*)',
        'last_name': r'Last Name\s*=\s*(.*)',
        'visit_date': r'Visit Date\s*=\s*(.*)',
        'test_name': r'Test Name\s*=\s*(.*)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content_str, re.IGNORECASE)
        if match: info[key] = match.group(1).strip()
    
    full = [info.get('first_name'), info.get('last_name')]
    info['full_name'] = " ".join(filter(None, full)) or "Unknown Patient"

    fs_hz = 20000.0 # Mặc định
    # FIX: Regex linh hoạt hơn cho phần tần số lấy mẫu
    fs_match = re.search(r'Sampling Frequency\(kHz\)\s*=\s*([0-9.,]+)', content_str)
    if fs_match:
        try:
            fs_val = float(fs_match.group(1).replace(',', '.'))
            fs_hz = fs_val * 1000
        except: pass
    
    # FIX QUAN TRỌNG: 
    # 1. Bắt cả "Sweep Data" VÀ "Averaged Data"
    # 2. Thêm \s* để chấp nhận khoảng trắng thừa nếu có
    regex_data = r'(?:Sweep|Averaged)\s*Data\(mV\)<(\d+)>=([\s\S]*?)(?=\[|$)'
    matches = list(re.finditer(regex_data, content_str))
    
    all_data = []
    boundaries = []
    current_idx = 0
    dt_ms = 1000.0 / fs_hz
    
    for i, match in enumerate(matches):
        raw_text_block = match.group(2)
        
        vals = extract_numbers_robust(raw_text_block)
        
        # Convert mV -> µV
        vals = vals * 1000 
        
        if len(vals) == 0: continue

        n_points = len(vals)
        boundaries.append({
            "trace_id": f"Trace {i+1}",
            "start_ms": current_idx * dt_ms,
            "end_ms": (current_idx + n_points) * dt_ms,
            "start_idx": current_idx,
            "end_idx": current_idx + n_points
        })
        all_data.append(vals)
        current_idx += n_points

    full_signal = np.concatenate(all_data) if all_data else np.array([])
    
    return {
        "data": full_signal,
        "fs_hz": fs_hz,
        "boundaries": boundaries,
        "patient_info": info,
        "metadata": {"source": "NATUS", "unit": "µV", "original_fs": fs_hz}
    }

# --- B. XỬ LÝ CSV ---
def validate_csv_buffer(decoded_bytes: bytes) -> bool:
    try:
        df = pd.read_csv(io.BytesIO(decoded_bytes), nrows=5)
        if df.empty: 
            return False
        is_numeric = df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())
        return is_numeric.any()
    except:
        return False

def load_csv_generic(file_path):
    try:
        df = pd.read_csv(file_path, header=None)
        try:
            pd.to_numeric(df.iloc[0, 0])
        except:
            df = pd.read_csv(file_path)

        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty: 
            return None

        DEFAULT_FS = 1000.0 
        dt_ms = 1000.0 / DEFAULT_FS
        
        all_data = []
        boundaries = []
        current_idx = 0
        
        for col_name in df_numeric.columns:
            vals = df_numeric[col_name].fillna(0).values * 1000 
            n_points = len(vals)
            boundaries.append({
                "trace_id": str(col_name),
                "start_ms": current_idx * dt_ms,
                "end_ms": (current_idx + n_points) * dt_ms,
                "start_idx": current_idx,
                "end_idx": current_idx + n_points
            })
            all_data.append(vals)
            current_idx += n_points
            
        full_signal = np.concatenate(all_data) if all_data else np.array([])
        
        display_name = "Unknown" 
        
        return {
            "data": full_signal,
            "fs_hz": DEFAULT_FS,
            "boundaries": boundaries,
            "patient_info": {
                "patient_id": "CSV_IMPORT",
                "full_name": display_name,
                "visit_date": "N/A",
                "test_name": "CSV Multi-Channel"
            },
            "metadata": {"source": "CSV", "num_channels": len(df_numeric.columns)}
        }
    except Exception as e:
        print(f"Lỗi load CSV: {e}")
        return None

# --- C. DISPATCHER ---
def load_data_from_file(file_path):
    path = Path(file_path)
    
    # 1. Check CSV
    if path.suffix.lower() == '.csv':
        return load_csv_generic(path)
            
    # 2. Check Natus TXT
    encodings_to_try = ['utf-16', 'utf-8', 'cp1258', 'latin1']
    
    content = None
    for enc in encodings_to_try:
        try:
            content = path.read_text(encoding=enc)
            if validate_natus_structure(content):
                return parse_natus_content(content)
        except:
            continue
    
    if content and validate_natus_structure(content):
         return parse_natus_content(content)

    return None

# --- D. HELPER FUNCTIONS  ---
def get_data_slice(file_path, start_ms, end_ms):
    result = load_data_from_file(file_path)
    if not result: return np.array([]), np.array([])
    
    full_signal = result['data']
    fs = result['fs_hz']
    dt_ms = 1000.0 / fs
    
    start_idx = int(start_ms / dt_ms)
    end_idx = int(end_ms / dt_ms)
    start_idx = max(0, start_idx)
    end_idx = min(len(full_signal), end_idx)
    
    if start_idx >= end_idx: return np.array([]), np.array([])
    return np.linspace(start_ms, end_ms, end_idx-start_idx), full_signal[start_idx:end_idx]

def get_downsampled_data(file_path, max_points=5000):
    result = load_data_from_file(file_path)
    if not result: return [], [], []
    full_signal = result['data']
    boundaries = result['boundaries']
    fs = result['fs_hz']
    dt_ms = 1000.0 / fs
    
    total_len = len(full_signal)
    step = max(1, total_len // max_points)
    ds_y = full_signal[::step]
    ds_x = np.linspace(0, total_len * dt_ms, len(ds_y))
    return ds_x, ds_y, boundaries