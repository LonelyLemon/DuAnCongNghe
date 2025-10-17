import datetime
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.widgets import SpanSelector
from pathlib import Path



ROOT = Path(__file__).resolve().parents[0]

PROCESSED_DATA_DIR = ROOT / "data_processed"
LABELS_DIR = ROOT / "labels"
PATIENTS_DIR = ROOT / "patients"

def append_label(patient_id, trace_id, start_s, end_s, label, note):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    path = LABELS_DIR / "labels.csv"
    row = pd.DataFrame([{
        "patient_id": patient_id,
        "trace_id": int(trace_id),
        "start_s": float(min(start_s, end_s)),
        "end_s": float(max(start_s, end_s)),
        "label": label,
        "note": note,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }])
    if path.exists(): row.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else: row.to_csv(path, index=False, encoding="utf-8")
    print("Saved label to:", path)

def pick_one_patient_id():
    js = sorted(PATIENTS_DIR.glob("*.json"))
    if not js: return "unknown"
    return js[0].stem

def main():
    csv_path = PROCESSED_DATA_DIR / "trace_data.csv"
    if not csv_path.exists():
        raise SystemExit("No processed data. Run parse_emg.py first.")
    df = pd.read_csv(csv_path)
    patient_id = pick_one_patient_id()
    trace_id = int(df["trace_id"].min())

    d = df[df["trace_id"]==trace_id]
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(d["time_s"], d["emg_mv"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG (mV)")
    ax.set_title(f"Patient {patient_id} — Trace {trace_id}")
    sel = {"x0": None, "x1": None}

    def onselect(xmin, xmax):
        sel["x0"], sel["x1"] = float(xmin), float(xmax)
        ax.axvspan(min(xmin, xmax), max(xmin, xmax), alpha=0.2)
        fig.canvas.draw_idle()

    selector = SpanSelector(
        ax, onselect, "horizontal",
        useblit=False,
        interactive=True,
        button=1,
        minspan=1e-6,
        props=dict(facecolor="tab:blue", alpha=0.2, edgecolor="none")
    )

    if hasattr(fig.canvas, "toolbar") and fig.canvas.toolbar is not None:
        try:
            fig.canvas.toolbar.pan(False)
            fig.canvas.toolbar.zoom(False)
        except Exception:
            pass

    plt.show()

    if sel["x0"] is not None and sel["x1"] is not None:
        label = input("Label (e.g., Benh A, Benh B): ").strip()
        note = input("Note: ").strip()
        append_label(patient_id, trace_id, sel["x0"], sel["x1"], label, note)
    else:
        print("No selection made.")

if __name__ == "__main__":
    main()
