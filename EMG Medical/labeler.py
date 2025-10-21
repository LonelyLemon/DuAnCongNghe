import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

ROOT = Path(__file__).resolve().parents[0]
PROCESSED = ROOT / "data_processed"
LABELS_DIR = ROOT / "labels"


Interval = Tuple[float, float]

def _normalize_interval(a: float, b: float) -> Interval:
    return (min(a, b), max(a, b))

def _merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals: return []
    ivs = sorted(intervals, key=lambda x: x[0])
    merged = [ivs[0]]
    for cur in ivs[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], cur[1]))
        else:
            merged.append(cur)
    return merged

def _subtract_interval(base: Interval, cut: Interval) -> List[Interval]:
    """Return base minus overlap with cut, can be 0/1/2 intervals."""
    a, b = base; c, d = cut
    if d <= a or c >= b:
        return [base]
    res = []
    if c > a: res.append((a, c))
    if d < b: res.append((d, b))
    return res

def toggle_interval(selected: List[Interval], new_iv: Interval) -> List[Interval]:
    new_iv = _normalize_interval(*new_iv)
    out: List[Interval] = []
    overlapped = False
    for iv in selected:
        parts = _subtract_interval(iv, new_iv)
        if len(parts) != 1 or parts[0] != iv:
            overlapped = True
        out.extend(parts)
    if not overlapped:
        out.append(new_iv)
    return _merge_intervals(out)


def save_labels(rows: List[dict]):
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    path = LABELS_DIR / "labels.csv"
    df = pd.DataFrame(rows)
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {len(rows)} segment(s) to:", path)


def run_labeler():
    csv_path = PROCESSED / "trace_data.csv"
    if not csv_path.exists():
        raise SystemExit("Missing data_processed/trace_data.csv. Run parse step first.")
    df = pd.read_csv(csv_path)

    needed = ["time_s","emg_mv","trace_id","patient_id","patient_name",
              "visit_datetime","tool_type","test_name","muscle"]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}' in trace_data.csv. Please include it at parse step.")

    trace_ids = sorted(df["trace_id"].unique().tolist())
    idx = 0
    active_label = "Benh A"
    pending_rows: List[dict] = []

    def show_trace(i: int):
        plt.close("all")
        t_id = trace_ids[i]
        d = df[df["trace_id"] == t_id].copy()

        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(d["time_s"], d["emg_mv"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EMG (mV)")

        pid = d["patient_id"].iloc[0]
        pname = d["patient_name"].iloc[0]
        vtime = d["visit_datetime"].iloc[0]
        tool = d["tool_type"].iloc[0]
        test = d["test_name"].iloc[0]
        muscle = d["muscle"].iloc[0]

        head1 = f"Patient: {pid}  {pname}    Visit: {vtime}"
        head2 = f"{tool}, {test} {muscle}"
        fig.suptitle(head1 + "\n" + head2, y=0.98, fontsize=11)

        _draw_scale_bar(ax, time_ms=10, amp_uv=500)

        
        sel_intervals = []
        last_span = None

        
        def onselect(xmin, xmax):
            nonlocal sel_intervals, last_span
            if last_span is not None:
                try:
                    last_span.remove()
                except Exception:
                    pass
                last_span = None
            new_iv = (float(xmin), float(xmax))
            sel_intervals = toggle_interval(sel_intervals, new_iv)
            _draw_intervals(ax, sel_intervals)
            fig.canvas.draw_idle()

        SpanSelector(ax, onselect, "horizontal",
                     interactive=True, useblit=False, button=1,
                     minspan=1e-6,
                     props=dict(facecolor="tab:blue", alpha=0.2, edgecolor="none"))

        def on_key(event):
            nonlocal active_label, pending_rows, sel_intervals, idx
            if event.key == "l":
                val = input(f"Active label (current='{active_label}'): ").strip()
                if val:
                    active_label = val
                    print("Active label:", active_label)
            elif event.key == "u":
                if sel_intervals:
                    sel_intervals.pop(-1)
                    _draw_intervals(ax, sel_intervals)
                    fig.canvas.draw_idle()
            elif event.key == "c":
                sel_intervals = []
                _draw_intervals(ax, sel_intervals)
                fig.canvas.draw_idle()
            elif event.key == "s":
                if not sel_intervals:
                    print("No segments to save.")
                    return
                now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
                meta = {
                    "patient_id": pid,
                    "trace_id": int(t_id),
                    "label": active_label,
                    "note": "",
                    "tool_type": tool,
                    "test_name": test,
                    "muscle": muscle,
                    "visit_datetime": vtime,
                    "created_at": now,
                }
                rows = []
                for a, b in sel_intervals:
                    rows.append({**meta, "start_s": float(a), "end_s": float(b)})
                save_labels(rows)
                pending_rows.clear()
            elif event.key == "n":
                if idx < len(trace_ids) - 1:
                    idx += 1
                    show_trace(idx)
            elif event.key == "p":
                if idx > 0:
                    idx -= 1
                    show_trace(idx)
            elif event.key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        try:
            if hasattr(fig.canvas, "toolbar") and fig.canvas.toolbar:
                fig.canvas.toolbar.pan(False)
                fig.canvas.toolbar.zoom(False)
        except Exception:
            pass

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

    def _draw_intervals(ax, intervals: List[Interval]):
        [p.remove() for p in getattr(ax, "_shade_patches", []) if p in ax.patches]
        ax._shade_patches = []
        for a, b in intervals:
            patch = ax.axvspan(a, b, facecolor="tab:blue", alpha=0.25, edgecolor="none")
            ax._shade_patches.append(patch)

    def _draw_scale_bar(ax, time_ms=10, amp_uv=500):
        """Draws a 10 ms (0.01 s) by 500 µV (0.5 mV) scalebar at bottom-right."""
        import numpy as np
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        dt = time_ms / 1000.0
        dv = amp_uv / 1000.0


        x0 = xlim[1] - (xlim[1]-xlim[0]) * 0.18
        y0 = ylim[0] + (ylim[1]-ylim[0]) * 0.08
        ax.plot([x0, x0+dt], [y0, y0], linewidth=2)
        ax.plot([x0, x0], [y0, y0+dv], linewidth=2)
        ax.text(x0 + dt/2, y0, f"{time_ms}ms", va="top", ha="center", fontsize=9)
        ax.text(x0, y0 + dv/2, f"{amp_uv}µV", va="center", ha="right", fontsize=9)

   
    show_trace(idx)

if __name__ == "__main__":
    run_labeler()
