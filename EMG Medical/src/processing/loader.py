import re
import numpy as np
from pathlib import Path

# --- HELPERS FUNCTION ---
def _parse_number_list(raw_chunk: str):
    s = raw_chunk.replace("/", " ").strip()
    nums = re.findall(r"-?\d+[,\.]\d+|-?\d+", s)
    return [float(x.replace(",", ".")) for x in nums]

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

def load_natus_txt(file_path: Path):
    with open(file_path, "r", encoding="utf-16", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    text = "\n".join(lines)

    # 1. Metadata
    patient_info = {
        "patient_id": _extract_value(text, "Patient ID"),
        "first_name": _extract_value(text, "First Name"),
        "visit_date": _extract_value(text, "Visit Date"),
        "test_name": _extract_value(text, "Full Name"),
    }
    
    subsampled_khz = _find_numeric_value(text, "Subsampled", "kHz") or 19.2
    device_info = {
        "sampling_khz": _find_numeric_value(text, "Sampling Frequency", "kHz"),
        "subsampled_khz": subsampled_khz,
        "low_filter_hz": _find_numeric_value(text, "Low", "Hz"),
        "high_filter_khz": _find_numeric_value(text, "High", "kHz"),
    }

    # 2. Extract Sweeps
    SWEEP_HDR_RE = re.compile(r"Sweep\s+Data\(mV\)<(\d+)>=")
    sweeps = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SWEEP_HDR_RE.search(line)
        if m:
            first = line.split("=", 1)[1]
            chunk_parts = [first]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if SWEEP_HDR_RE.search(nxt) or nxt.startswith("["): break
                chunk_parts.append(nxt)
                i += 1
            sweeps.append("\n".join(chunk_parts))
            continue
        i += 1

    # 3. Process Traces & Concatenate Timeline
    full_voltage_list = []
    boundaries = []
    current_offset = 0.0
    sweep_dur = _find_numeric_value(text, "Sweep Duration", "ms") or 100.0
    dt_ms = 1_000.0 / (subsampled_khz * 1000.0)

    for idx, raw in enumerate(sweeps, 1):
        vals = np.array(_parse_number_list(raw), dtype=float) * 1000.0 # to uV
        
        real_dt = float(sweep_dur) / len(vals) if len(vals) else dt_ms
        
        trace_id = f"trace_{idx:03d}"
        full_voltage_list.extend(vals)
        
        end_time = current_offset + (len(vals) * real_dt)
        boundaries.append({
            "trace_id": trace_id, 
            "start_ms": current_offset, 
            "end_ms": end_time
        })
        current_offset = end_time

    return {
        "patient_info": patient_info,
        "device_info": device_info,
        "full_sequence": {
            "dt_ms": dt_ms,
            "boundaries": boundaries
        },
        "full_data_stream": full_voltage_list
    }