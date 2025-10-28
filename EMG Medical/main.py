import re
import json
import numpy as np
from pathlib import Path


# ----------- FILE PATH CONFIG -----------
BASE_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = BASE_DIR / "data_raw"
DATA_PROCESSED_DIR = BASE_DIR / "data_processed"

DATA_PROCESSED_DIR.mkdir(exist_ok=True)


# Input & Output paths
file_path = DATA_RAW_DIR / "Le Thi Bop.txt"
emg_data_path = DATA_PROCESSED_DIR / "emg_data.json"


# ----------- READ RAW DATA (UTF-16) -----------
with open(file_path, "r", encoding="utf-16", errors="ignore") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

text = "\n".join(lines)


# ----------- HELPERS -----------
SWEEP_HDR_RE = re.compile(r"Sweep\s+Data\(mV\)<(\d+)>=")
LONGTRACE_HDR_RE = re.compile(r"LongTrace\s+Data\(mV\)<(\d+)>=")

def _is_section_header(line: str) -> bool:
    return line.startswith("[") and line.endswith("]")

def _is_kv_metadata(line: str) -> bool:
    return bool(re.match(r"^[^0-9\-\s].*?=.+$", line))

def collect_sweeps(all_lines):
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

def collect_longtrace(all_lines):
    i = 0
    while i < len(all_lines):
        line = all_lines[i]
        m = LONGTRACE_HDR_RE.search(line)
        if m:
            n_samples = int(m.group(1))
            first = line.split("=", 1)[1]
            chunk_parts = [first]
            i += 1
            while i < len(all_lines):
                nxt = all_lines[i]
                if SWEEP_HDR_RE.search(nxt) or LONGTRACE_HDR_RE.search(nxt):
                    break
                if _is_section_header(nxt) or _is_kv_metadata(nxt):
                    break
                chunk_parts.append(nxt)
                i += 1
            return ("\n".join(chunk_parts), n_samples)
        i += 1
    return (None, 0)


def extract_value(key: str):
    pattern = rf"{re.escape(key)}=(.+)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def find_numeric_value(label: str, unit: str):
    pattern = rf"{re.escape(label)}\({re.escape(unit)}\)\s*=\s*([0-9.,]+|Off)"
    m = re.search(pattern, text)
    if not m:
        return None
    raw = m.group(1).strip()
    if raw.lower() == "off":
        return None
    return float(raw.replace(",", "."))


def parse_number_list(raw_chunk: str):
    s = raw_chunk.replace("/", " ").strip()
    nums = re.findall(r"-?\d+[,\.]\d+|-?\d+", s)
    return [float(x.replace(",", ".")) for x in nums]


# ----------- PATIENT INFO -----------
patient_info = {
    "patient_id": extract_value("Patient ID"),
    "first_name": extract_value("First Name"),
    "gender": extract_value("Gender"),
    "visit_date": extract_value("Visit Date"),
    "visit_type": extract_value("Visit Type"),
    "test_name": extract_value("Full Name"),
    "muscle_name": extract_value("Master Anatomy"),
}


# ----------- DEVICE INFO -----------
sampling_khz = find_numeric_value("Sampling Frequency", "kHz")
subsampled_khz = find_numeric_value("Subsampled", "kHz")
low_filter_hz = find_numeric_value("Low", "Hz")
high_filter_khz = find_numeric_value("High", "kHz")
notch_filter_hz = find_numeric_value("Notch Filter", "Hz")
amplifier_range_mv = find_numeric_value("Amplifier Range", "mV")

device_info = {
    "test_type": extract_value("Full Name"),
    "anatomy": extract_value("Master Anatomy"),
    "sampling_frequency_khz": sampling_khz,
    "subsampled_khz": subsampled_khz,
    "low_filter_hz": low_filter_hz,
    "high_filter_khz": high_filter_khz,
    "notch_filter_hz": notch_filter_hz,
    "amplifier_range_mv": amplifier_range_mv,
    "unit": "µV",
}


# ----------- TRACE DATA (ALL 200 TRACES) -----------
sweeps_raw = collect_sweeps(lines)
assert len(sweeps_raw) == 200, f"Expected 200 sweeps trace, got {len(sweeps_raw)}"

sweep_duration_ms = find_numeric_value("Sweep Duration", "ms") or 100.0

dt_ms = 1_000.0 / (subsampled_khz * 1000.0)

traces_dict = {}
for idx, (raw_chunk, n_declared) in enumerate(sweeps_raw, start=1):
    mv_values = np.array(parse_number_list(raw_chunk), dtype=float)
    if n_declared and n_declared != len(mv_values):
        raise ValueError(f"Sweep {idx}: header count {n_declared} != parsed {len(mv_values)}")

    dt_ms = float(sweep_duration_ms) / len(mv_values) if len(mv_values) else 0.0

    uv_values = mv_values * 1000.0
    time_ms = np.arange(len(uv_values), dtype=float) * dt_ms

    trace_id = f"trace_{idx:03d}"
    traces_dict[trace_id] = {
        "trace_meta": {
            "trace_index": idx,
            "num_samples": int(len(uv_values)),
            "dt_ms": dt_ms,
        },
        "trace_data": [
            {"time_ms": float(t), "voltage_uv": float(v)}
            for t, v in zip(time_ms, uv_values)
        ],
    }

# Parse LongTrace Data
long_raw, long_n = collect_longtrace(lines)
long_trace = None

if long_raw:
    long_mv = np.array(parse_number_list(long_raw), dtype=float)
    if long_n and long_n != len(long_mv):
        raise ValueError(f"LongTrace: header count {long_n} != parsed {len(long_mv)}")
    
    if subsampled_khz and subsampled_khz > 0:
        long_dt_ms = 1_000.0 / (subsampled_khz * 1000.0)
    else:
        long_dt_ms = 0.0

    long_time_ms = np.arange(len(long_mv), dtype=float) * long_dt_ms
    long_uv = long_mv * 1000.0

    long_trace = {
        "trace_meta": {
            "num_samples": int(len(long_uv)),
            "dt_ms": float(long_dt_ms),
        },
        "trace_data": [
            {"time_ms": float(t), "voltage_uv": float(v)}
            for t, v in zip(long_time_ms, long_uv)
        ],
    }

# ----------- SAVE EMG DATA -----------
emg_data = {
    "patient_info": patient_info,
    "device_info": device_info,
    "traces": traces_dict,
    "long_trace": long_trace
}

with open(emg_data_path, "w", encoding="utf-16") as f:
    json.dump(emg_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved EMG data to: {emg_data_path.name} (UTF-16)")
print(f"📊 Total traces: {len(sweeps_raw)}")
print("🎉 Processing completed successfully!")
