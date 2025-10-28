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


def collect_traces(all_lines):
    traces_raw = []
    i = 0
    header_re = re.compile(r"(?:Sweep\s+Data|Averaged\s+Data)\(mV\)<\d+>=")
    while i < len(all_lines):
        line = all_lines[i]
        if header_re.search(line):
            first = line.split("=", 1)[1]
            chunk_parts = [first]
            i += 1
            while i < len(all_lines):
                nxt = all_lines[i]
                if re.match(r".+=.+", nxt) and not re.match(r"^-?\d", nxt):
                    break
                if nxt.startswith("[") and nxt.endswith("]"):
                    break
                chunk_parts.append(nxt)
                i += 1
            traces_raw.append("\n".join(chunk_parts))
            continue
        i += 1
    return traces_raw


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
traces_raw = collect_traces(lines)


dt_ms = 1.0 / (sampling_khz * 1000.0) * 1000.0 if (sampling_khz and sampling_khz > 0) else 0.0208333333

traces_dict = {}
for idx, raw_chunk in enumerate(traces_raw, start=1):
    mv_values = np.array(parse_number_list(raw_chunk), dtype=float)
    uv_values = mv_values * 1000.0
    time_ms = np.arange(len(uv_values), dtype=float) * dt_ms

    trace_id = f"trace_{idx:03d}"
    traces_dict[trace_id] = {
        "trace_meta": {
            "trace_index": idx,
            "num_samples": int(len(uv_values)),
            "dt_ms": float(dt_ms),
        },
        "trace_data": [
            {"time_ms": float(t), "voltage_uv": float(v)}
            for t, v in zip(time_ms, uv_values)
        ],
    }


# ----------- SAVE EMG DATA -----------
emg_data = {
    "patient_info": patient_info,
    "device_info": device_info,
    "traces": traces_dict,
}

with open(emg_data_path, "w", encoding="utf-16") as f:
    json.dump(emg_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved EMG data to: {emg_data_path.name} (UTF-16)")
print(f"📊 Total traces: {len(traces_raw)}")
print("🎉 Processing completed successfully!")
