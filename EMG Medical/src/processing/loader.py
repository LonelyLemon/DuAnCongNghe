# File: src/processing/loader.py
import re
import numpy as np
import base64
import io
from pathlib import Path

# --- PHẦN 1: CÁC HÀM HELPERS (Khôi phục từ main.py gốc) ---
# Lý do: Logic này xử lý file Natus tốt hơn regex đơn thuần

SWEEP_HDR_RE = re.compile(r"Sweep\s+Data\(mV\)<(\d+)>=")
LONGTRACE_HDR_RE = re.compile(r"LongTrace\s+Data\(mV\)<(\d+)>=")

def _is_section_header(line: str) -> bool:
    """Kiểm tra xem dòng này có phải là header section [1.1...] không"""
    return line.startswith("[") and line.endswith("]")

def _is_kv_metadata(line: str) -> bool:
    """Kiểm tra xem dòng này có phải là metadata dạng Key=Value không"""
    return bool(re.match(r"^[^0-9\-\s].*?=.+$", line))

def _extract_value(text, key):
    pattern = rf"{re.escape(key)}=(.+)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None

def _find_numeric_value(text, label, unit):
    pattern = rf"{re.escape(label)}\({re.escape(unit)}\)\s*=\s*([0-9.,]+|Off)"
    m = re.search(pattern, text)
    if not m: return None
    raw = m.group(1).strip()
    return None if raw.lower() == "off" else float(raw.replace(",", "."))

def parse_number_list(raw_chunk: str):
    """
    Parse chuỗi chứa số liệu. 
    Code cũ xử lý tốt việc loại bỏ ký tự lạ và phân tách dấu phẩy/chấm.
    """
    s = raw_chunk.replace("/", " ").strip()
    nums = re.findall(r"-?\d+[,\.]\d+|-?\d+", s)
    return [float(x.replace(",", ".")) for x in nums]

def collect_sweeps(all_lines):
    """
    [QUAN TRỌNG] Logic thu thập Sweep thông minh.
    Nó sẽ dừng đọc chunk ngay khi gặp Header hoặc Metadata -> Tránh đọc nhầm rác.
    """
    sweeps = []
    i = 0
    while i < len(all_lines):
        line = all_lines[i]
        m = SWEEP_HDR_RE.search(line)
        if m:
            n_samples = int(m.group(1))
            first = line.split("=", 1)[1]
            chunk_parts = [first]
            i += 1
            while i < len(all_lines):
                nxt = all_lines[i]
                # Điều kiện dừng quan trọng:
                if SWEEP_HDR_RE.search(nxt) or LONGTRACE_HDR_RE.search(nxt):
                    break
                if _is_section_header(nxt) or _is_kv_metadata(nxt):
                    break
                chunk_parts.append(nxt)
                i += 1
            sweeps.append(("\n".join(chunk_parts), n_samples))
            continue
        i += 1
    return sweeps

# --- PHẦN 2: LOGIC XỬ LÝ CHÍNH (Kết hợp cả đọc file và parse nội dung upload) ---

def parse_natus_content(content_str: str):
    """
    Hàm xử lý nội dung text (Dùng chung cho cả File Load và Upload)
    """
    # Tách dòng
    lines = [line.strip() for line in content_str.splitlines() if line.strip()]
    text = "\n".join(lines)

    # 1. Metadata
    patient_info = {
        "patient_id": _extract_value(text, "Patient ID"),
        "first_name": _extract_value(text, "First Name"),
        "visit_date": _extract_value(text, "Visit Date"),
        "test_name": _extract_value(text, "Full Name"),
    }
    
    # 2. Device Info & Time
    subsampled_khz = _find_numeric_value(text, "Subsampled", "kHz")
    if not subsampled_khz:
        samp_freq = _find_numeric_value(text, "Sampling Frequency", "kHz")
        subsampled_khz = samp_freq if samp_freq else 19.2
        
    dt_ms = 1_000.0 / (subsampled_khz * 1000.0)
    sweep_dur = _find_numeric_value(text, "Sweep Duration", "ms") or 100.0

    # 3. Extract Sweeps (DÙNG LOGIC CŨ ĐỂ CHUẨN XÁC)
    sweeps_raw = collect_sweeps(lines)

    full_voltage_list = []
    boundaries = []
    current_offset = 0.0

    for idx, (raw_chunk, n_declared) in enumerate(sweeps_raw, 1):
        mv_values = np.array(parse_number_list(raw_chunk), dtype=float)
        
        # Validation nhẹ
        if n_declared and n_declared != len(mv_values):
            pass 

        uv_values = mv_values * 1000.0
        
        # Tính thời gian thực tế dựa trên số điểm
        real_duration = len(uv_values) * dt_ms
        
        full_voltage_list.extend(uv_values)
        
        boundaries.append({
            "trace_id": f"trace_{idx:03d}", 
            "start_ms": current_offset, 
            "end_ms": current_offset + real_duration
        })
        
        # Để timeline liền mạch theo đúng Sweep Duration (thường là 100ms)
        step = max(real_duration, float(sweep_dur))
        current_offset += step

    return {
        "patient_info": patient_info,
        "full_sequence": {
            "dt_ms": dt_ms,
            "boundaries": boundaries
        },
        "full_data_stream": full_voltage_list
    }

def load_natus_txt(file_path: Path):
    """
    Wrapper đọc file từ ổ cứng
    """
    try:
        with open(file_path, "r", encoding="utf-16", errors="ignore") as f:
            content = f.read()
        return parse_natus_content(content)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

# --- PHẦN 3: CÁC HÀM SLICING PHỤC VỤ DASH APP (Lazy Loading) ---

def get_data_slice(file_path, start_ms, end_ms):
    """Lấy dữ liệu High-Res cho một đoạn"""
    data = load_natus_txt(file_path)
    full_stream = data.get("full_data_stream", [])
    dt_ms = data.get("full_sequence", {}).get("dt_ms", 0.02)
    
    if not full_stream: return np.array([]), np.array([])
    
    start_idx = int(start_ms / dt_ms)
    end_idx = int(end_ms / dt_ms)
    
    # Slice an toàn
    vals = np.array(full_stream[max(0, start_idx) : min(len(full_stream), end_idx)], dtype=float)
    # Tính lại trục thời gian chuẩn xác từ start_idx
    times = np.arange(len(vals), dtype=float) * dt_ms + (start_idx * dt_ms)
    
    return times, vals

def get_downsampled_data(file_path, max_points=5000):
    """Lấy dữ liệu Low-Res cho Overview"""
    data = load_natus_txt(file_path)
    full_stream = data.get("full_data_stream", [])
    dt_ms = data.get("full_sequence", {}).get("dt_ms", 0.02)
    boundaries = data.get("full_sequence", {}).get("boundaries", [])
    
    if not full_stream: return np.array([]), np.array([]), []

    full_vals = np.array(full_stream, dtype=float)
    
    # Downsample
    step = max(1, len(full_vals) // max_points)
    ds_vals = full_vals[::step]
    ds_times = np.arange(len(ds_vals), dtype=float) * dt_ms * step
    
    return ds_times, ds_vals, boundaries