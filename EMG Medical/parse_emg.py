import re, json
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

RAW_DATA_DIR = ROOT / "data_raw"
PROCESSED_DATA_DIR = ROOT / "data_processed"
PATIENTS_DATA_DIR = ROOT / "patients"

# Make directory for data
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PATIENTS_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


NUM_RE = re.compile(r'[+-]?\d+,\d+')
def to_float_comma(x: str) -> float:
    return float(x.replace(",", "."))


# Data processing step
def parse_emg_txt(path: Path):
    text = path.read_text(encoding="utf-16", errors="ignore")
    lines = text.splitlines()

    patient_info = {}
    meta = {}
    muscle = None
    visit_datetime = None
    freq_smpl = None

    traces = []
    trace_id = 0

    for line in lines:
        s = line.strip()

        # Patient Info
        if s.startswith("Patient ID="): 
            patient_info["patient_id"] = s.split("=", 1)[1].strip()
        elif s.startswith("First Name="): 
            patient_info["first_name"] = s.split("=",1)[1].strip()
        elif s.startswith("Gender="): 
            patient_info["gender"] = s.split("=",1)[1].strip()

        # Visit Date
        elif s.startswith("Visit Date="): 
            visit_datetime = s.split("=",1)[1].strip()

        # Anatomy 
        elif s.startswith("Master Anatomy="): 
            muscle = s.split("=",1)[1].strip()

        # Sampling parameters
        elif s.startswith("Subsampled(kHz)="): 
            meta["Subsampled(kHz)"] = s.split("=",1)[1].strip()
        elif s.startswith("Sampling Frequency(kHz)="): 
            meta["Sampling Frequency(kHz)"] = s.split("=",1)[1].strip()
        elif s.startswith("Sweep Duration(ms)="): 
            meta["Sweep Duration(ms)"] = s.split("=",1)[1].strip()

        elif s.startswith("Test="): 
            meta["Test"] = s.split("=",1)[1].strip()
        elif s.startswith("Full Name="): 
            meta["Full Name"] = s.split("=",1)[1].strip()
        elif s.startswith("Low(Hz)="): 
            meta["Low(Hz)"] = s.split("=",1)[1].strip()
        elif s.startswith("High(kHz)="): 
            meta["High(kHz)"] = s.split("=",1)[1].strip()
        elif s.startswith("Amplifier Range(mV)="): 
            meta["Amplifier Range(mV)"] = s.split("=",1)[1].strip()


        # Trace
        elif s.startswith("Sweep  Data(mV)<"):
            trace_id += 1
            nums = NUM_RE.findall(s)
            traces.append((trace_id, [to_float_comma(x) for x in nums]))
        else:
            if trace_id>0 and NUM_RE.search(s):
                traces[-1][1].extend([to_float_comma(x) for x in NUM_RE.findall(s)])

    # Frequency Sampling (Hz)
    
    if "Subsampled(kHz)" in meta:
        try:
            freq_smpl = to_float_comma(meta["Subsampled(kHz)"]) * 1000.0
        except:
            pass
    if not freq_smpl and "Sampling Frequency(kHz)" in meta:
        try:
            freq_smpl = to_float_comma(meta["Sampling Frequency(kHz)"]) * 1000.0
        except:
            pass
    
    frames = []
    for trace_id, vals in traces:
        N = len(vals)
        time_s = np.arange(N, dtype=float)/freq_smpl if freq_smpl and freq_smpl > 0 else np.arange(N, dtype=float)
        frames.append(pd.DataFrame({
            "time_s": time_s,
            "emg_mv": np.array(vals, dtype=float),
            "trace_id": trace_id,
            "muscle": muscle,
            "visit_datetime": visit_datetime,
        }))
    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # patient info
    patient_info.setdefault("patient_id","unknown")
    patient_info["muscle"] = muscle
    patient_info["visit_datetime"] = visit_datetime
    if freq_smpl: 
        patient_info["sampling_hz"] = freq_smpl
    return patient_info, meta, df_all

def main():
    candidates = sorted(RAW_DATA_DIR.glob("*.txt"))
    if not candidates:
        raise SystemExit("Put EMG .txt files into data_raw/ then rerun.")
    path = candidates[0]
    patient, meta, df_all = parse_emg_txt(path)

    print("Patient:", patient)
    print("Meta(head):", dict(list(meta.items())[:3]))
    
    PATIENTS_DATA_DIR.joinpath(f"{patient.get('patient_id','unknown')}.json").write_text(
        json.dumps(patient, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not df_all.empty:
        out = PROCESSED_DATA_DIR / "trace_data.csv"
        df_all.to_csv(out, index=False)
        print("Saved:", out)

if __name__ == "__main__":
    main()