import json
import time
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, Tuple, List

from dash import (
    Dash, 
    dcc, 
    html, 
    Input, 
    Output, 
    State,
    dash_table,
    no_update, 
    ctx)


BASE_DIR = Path(__file__).resolve().parent
EMG_JSON = BASE_DIR / "data_processed" / "emg_data.json"
LABELED_DIR = BASE_DIR / "labeled_data"
LABELED_DIR.mkdir(exist_ok=True)


# ---------- Loading processed data ----------
with open(EMG_JSON, "r", encoding="utf-16") as f:
    emg = json.load(f)

DEVICE = emg.get("device_info", {})
PATIENT = emg.get("patient_info", {}) or {}
TRACES: Dict[str, Any] = emg.get("traces", {})
LONG: Dict[str, Any] = emg.get("long_trace", None)
TRACE_IDS = sorted(TRACES.keys())


def _xy_from_trace_dict(t: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    dt_ms = float(t.get("trace_meta", {}).get("dt_ms", 0.0208333333))
    arr = t.get("trace_data", [])
    time_ms = np.fromiter((p["time_ms"] for p in arr), dtype=float, count=len(arr))
    volt_uv = np.fromiter((p["voltage_uv"] for p in arr), dtype=float, count=len(arr))
    return time_ms, volt_uv, dt_ms


def get_trace_xy(trace_id: str) -> Tuple[np.ndarray, np.ndarray, float]:
    if trace_id == "long_trace":
        if not LONG:
            return np.array([]), np.array([]), 0.0
        return _xy_from_trace_dict(LONG)
    t = TRACES[trace_id]
    return _xy_from_trace_dict(t)


def downsample_stride(x: np.ndarray, y: np.ndarray, max_points: int = 120_000):
    n = x.size
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    return x[::step], y[::step]


def initial_window(x: np.ndarray, width_ms: float) -> Tuple[float, float]:
    if x.size == 0:
        return (0.0, width_ms)
    left = float(x[0])
    right = float(min(x[-1], left + width_ms))
    return (left, right)


def extract_all_rects(relayout: dict) -> List[Tuple[float, float]]:
    rects: List[Tuple[float, float]] = []
    if not relayout:
        return rects
    if "shapes" in relayout and isinstance(relayout["shapes"], list):
        for shp in relayout["shapes"]:
            if isinstance(shp, dict) and ("x0" in shp and "x1" in shp):
                x0, x1 = float(shp["x0"]), float(shp["x1"])
                if x1 < x0:
                    x0, x1 = x1, x0
                rects.append((x0, x1))
        return rects
    indices = set()
    for k in relayout.keys():
        if k.startswith("shapes[") and (k.endswith("].x0") or k.endswith("].x1")):
            idx = int(k.split("[")[-1].split("]")[0])
            indices.add(idx)
    for idx in sorted(indices):
        kx0 = f"shapes[{idx}].x0"
        kx1 = f"shapes[{idx}].x1"
        if kx0 in relayout and kx1 in relayout:
            x0, x1 = float(relayout[kx0]), float(relayout[kx1])
            if x1 < x0:
                x0, x1 = x1, x0
            rects.append((x0, x1))
    return rects


def patient_rows() -> List[Dict[str, str]]:
    p = PATIENT or {}
    return [
        {"Field": "Patient ID", "Value": p.get("patient_id", "")},
        {"Field": "First Name", "Value": p.get("first_name", "")},
        {"Field": "Gender", "Value": p.get("gender", "")},
        {"Field": "Visit Date", "Value": p.get("visit_date", "")},
        {"Field": "Visit Type", "Value": p.get("visit_type", "")},
        {"Field": "Test Name", "Value": p.get("test_name", "")},
        {"Field": "Muscle", "Value": p.get("muscle_name", "")},
    ]


def device_rows() -> List[Dict[str, str]]:
    d = DEVICE or {}
    return [
        {"Field": "Test Type", "Value": d.get("test_type", "")},
        {"Field": "Anatomy", "Value": d.get("anatomy", "")},
        {"Field": "Sampling Frequency (kHz)", "Value": str(d.get("sampling_frequency_khz", ""))},
        {"Field": "Subsampled (kHz)", "Value": str(d.get("subsampled_khz", ""))},
        {"Field": "Low Filter (Hz)", "Value": str(d.get("low_filter_hz", ""))},
        {"Field": "High Filter (kHz)", "Value": str(d.get("high_filter_khz", ""))},
        {"Field": "Notch Filter (Hz)", "Value": str(d.get("notch_filter_hz", ""))},
        {"Field": "Amplifier Range (mV)", "Value": str(d.get("amplifier_range_mv", ""))},
        {"Field": "Unit", "Value": d.get("unit", "µV")},
    ]

app = Dash(__name__)
app.title = "EMG Data Visualization and Labeling"

patient_table = dash_table.DataTable(
    id="patient-table",
    columns=[{"name": "Field", "id": "Field"}, {"name": "Value", "id": "Value"}],
    data=patient_rows(),
    style_cell={"padding": "6px", "fontFamily": "system-ui, Arial", "fontSize": 14},
    style_header={"fontWeight": "bold"},
    style_table={"maxHeight": "230px", "overflowY": "auto"},
)

device_table = dash_table.DataTable(
    id="device-table",
    columns=[{"name": "Field", "id": "Field"}, {"name": "Value", "id": "Value"}],
    data=device_rows(),
    style_cell={"padding": "6px", "fontFamily": "system-ui, Arial", "fontSize": 14},
    style_header={"fontWeight": "bold"},
    style_table={"maxHeight": "260px", "overflowY": "auto"},
)

labels_table = dash_table.DataTable(
    id="labels-table",
    columns=[
        {"name": "start time (ms)", "id": "start_ms", "type": "numeric"},
        {"name": "end time (ms)", "id": "end_ms", "type": "numeric"},
        {"name": "trace_id", "id": "trace_id", "type": "text"},
        {"name": "label", "id": "label", "presentation": "dropdown"},
    ],
    data=[],
    editable=True,
    row_deletable=True,
    dropdown={
        "label": {
            "options": [
                {"label": "bệnh A", "value": "bệnh A"},
                {"label": "bệnh B", "value": "bệnh B"},
                {"label": "unknown", "value": "unknown"},
            ]
        }
    },
    style_cell={"padding": "6px", "fontFamily": "system-ui, Arial", "fontSize": 14},
    style_header={"fontWeight": "bold"},
    style_table={"maxHeight": "300px", "overflowY": "auto", "border": "1px solid #e5e7eb"},
)

app.layout = html.Div(
    style={"fontFamily": "system-ui, Arial", "padding": "14px", "maxWidth": "1280px", "margin": "0 auto"},
    children=[
        html.H2("EMG Trace Labeler (Plotly + Dash)"),

        html.Div([
            html.Div([html.H4("Patient Info"), patient_table],
                     style={"flex": "1", "minWidth": "340px", "marginRight": "16px"}),
            html.Div([html.H4("Device / Muscle Info"), device_table],
                     style={"flex": "1", "minWidth": "360px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "8px"}),

        html.Hr(),

        html.Div([
            # Sweep controls
            html.Div([
                html.H4("Sweep Traces (200)"),
                html.Label("Trace"),
                dcc.Dropdown(
                    id="trace-dd",
                    options=[{"label": tid, "value": tid} for tid in TRACE_IDS],
                    value=TRACE_IDS[0] if TRACE_IDS else None,
                    clearable=False,
                ),
                html.Div([
                    html.Label("Window (ms)"),
                    dcc.Input(id="win-ms", type="number", value=500.0, min=10, step=10),
                ], style={"marginTop": "6px"}),
                html.Div([
                    html.Button("⟵ Pan", id="pan-left", n_clicks=0, style={"marginRight": "6px"}),
                    html.Button("Pan ⟶", id="pan-right", n_clicks=0),
                ], style={"marginTop": "6px"}),
            ], style={"flex": "1", "minWidth": "320px", "marginRight": "12px"}),

            # LongTrace controls
            html.Div([
                html.H4("LongTrace (full session)"),
                html.Div([
                    html.Label("Window (ms)"),
                    dcc.Input(id="win-ms-long", type="number", value=5000.0, min=10, step=10),
                ]),
                html.Div([
                    html.Button("⟵ Pan (Long)", id="pan-left-long", n_clicks=0, style={"marginRight": "6px"}),
                    html.Button("Pan ⟶ (Long)", id="pan-right-long", n_clicks=0),
                ], style={"marginTop": "6px"}),
            ], style={"flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),

        # Two parallel charts
        html.Div([
            dcc.Graph(
                id="trace-graph",
                figure=go.Figure(),
                style={"height": "60vh"},
                config={"displaylogo": False, "modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
            ),
            dcc.Graph(
                id="long-graph",
                figure=go.Figure(),
                style={"height": "60vh"},
                config={"displaylogo": False, "modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
            ),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "12px"}),

        html.Div(id="selection-info", style={"marginTop": "8px", "fontSize": 14, "color": "#374151"}),

        # Label actions
        html.Div([
            html.Button("Add Sweep selections to table", id="add-selection-sweep", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Add LongTrace selections to table", id="add-selection-long", n_clicks=0, style={"marginRight": "12px"}),
            html.Button("Erase selections (Sweep)", id="erase-selection-sweep", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Erase selections (Long)", id="erase-selection-long", n_clicks=0),
        ], style={"marginTop": "10px"}),

        html.Div([
            html.Button("Save labels", id="save-labels", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Clear labels", id="clear-labels", n_clicks=0),
            html.Span(id="save-status", style={"marginLeft": "12px", "color": "#047857"}),
        ], style={"marginTop": "10px"}),

        html.H4("Labels"),
        labels_table,

        # stores
        dcc.Store(id="xrange-store"),   # Sweep Trace
        dcc.Store(id="rects-store-sweep"),
        
        dcc.Store(id="xrange-store-long"),  # Long Trace
        dcc.Store(id="rects-store-long"),
    ]
)


# ---------------- Callbacks ----------------
@app.callback(
    Output("trace-graph", "figure"),
    Output("xrange-store", "data"),
    Input("trace-dd", "value"),
    Input("win-ms", "value"),
    Input("pan-left", "n_clicks"),
    Input("pan-right", "n_clicks"),
    State("xrange-store", "data"),
    prevent_initial_call=False
)
def update_plot(trace_id, win_ms, n_left, n_right, xrange_mem):
    if not trace_id or not win_ms:
        return no_update, no_update

    x, y, _ = get_trace_xy(trace_id)
    x_ds, y_ds = downsample_stride(x, y)

    if xrange_mem and xrange_mem.get("trace_id") == trace_id:
        x0, x1 = float(xrange_mem.get("x0")), float(xrange_mem.get("x1"))
    else:
        x0, x1 = initial_window(x, float(win_ms))

    pan_step = float(win_ms) / 2.0
    trigger = ctx.triggered_id
    if trigger == "pan-left" and n_left:
        x0 = max(float(x[0]), x0 - pan_step)
        x1 = x0 + float(win_ms)
    elif trigger == "pan-right" and n_right:
        x1 = min(float(x[-1]), x1 + pan_step)
        x0 = x1 - float(win_ms)

    if x.size > 0:
        if x0 < float(x[0]):
            x0 = float(x[0]); x1 = x0 + float(win_ms)
        if x1 > float(x[-1]):
            x1 = float(x[-1]); x0 = x1 - float(win_ms)
            if x0 < float(x[0]):
                x0 = float(x[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name=trace_id))
    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (µV)",
        xaxis=dict(rangeslider=dict(visible=True), range=[x0, x1]),
        template="plotly_white",
        dragmode="pan",
        shapes=[],
    )
    store = {"trace_id": trace_id, "x0": x0, "x1": x1}
    return fig, store

# -------- Long figure (parallel to sweep) --------
@app.callback(
    Output("long-graph", "figure"),
    Output("xrange-store-long", "data"),
    Input("win-ms-long", "value"),
    Input("pan-left-long", "n_clicks"),
    Input("pan-right-long", "n_clicks"),
    State("xrange-store-long", "data"),
    prevent_initial_call=False
)
def update_plot_long(win_ms, n_left, n_right, xrange_mem):
    trace_id = "long_trace"
    x, y, _ = get_trace_xy(trace_id)
    x_ds, y_ds = downsample_stride(x, y)

    if x.size == 0:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Time (ms)",
            yaxis_title="Voltage (µV)",
            template="plotly_white"
        )
        return fig, None

    if xrange_mem:
        x0, x1 = float(xrange_mem.get("x0")), float(xrange_mem.get("x1"))
    else:
        x0, x1 = initial_window(x, float(win_ms or 5000.0))

    pan_step = float(win_ms or 5000.0) / 2.0
    trigger = ctx.triggered_id
    if trigger == "pan-left-long" and n_left:
        x0 = max(float(x[0]), x0 - pan_step); x1 = x0 + float(win_ms)
    elif trigger == "pan-right-long" and n_right:
        x1 = min(float(x[-1]), x1 + pan_step); x0 = x1 - float(win_ms)

    if x0 < float(x[0]): x0, x1 = float(x[0]), float(x[0]) + float(win_ms)
    if x1 > float(x[-1]): x1, x0 = float(x[-1]), float(x[-1]) - float(win_ms); x0 = max(x0, float(x[0]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode="lines", name="long_trace"))
    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Time (ms)",
        yaxis_title="Voltage (µV)",
        xaxis=dict(rangeslider=dict(visible=True), range=[x0, x1]),
        template="plotly_white",
        dragmode="pan",
        shapes=[],
    )
    store = {"trace_id": trace_id, "x0": x0, "x1": x1}
    return fig, store


# Sweep selections
@app.callback(
    Output("rects-store-sweep", "data"),
    Output("selection-info", "children"),
    Input("trace-graph", "relayoutData"),
    prevent_initial_call=False
)
def on_relayout_sweep(relayout):
    rects = extract_all_rects(relayout) if relayout else []
    if not rects:
        return [], "Tip: Use 'Draw rectangle' on either chart, then press the corresponding 'Add ... selections' button."
    return rects, f"{len(rects)} selection(s) ready on Sweep chart."

# Long selections
@app.callback(
    Output("rects-store-long", "data"),
    Input("long-graph", "relayoutData"),
    prevent_initial_call=False
)
def on_relayout_long(relayout):
    rects = extract_all_rects(relayout) if relayout else []
    return rects


@app.callback(
    Output("labels-table", "data"),
    Input("add-selection-sweep", "n_clicks"),
    State("labels-table", "data"),
    State("rects-store-sweep", "data"),
    State("trace-dd", "value"),
)
def add_sel_sweep(n_clicks, rows, rects, trace_id):
    if not n_clicks:
        return no_update
    if not rects or not trace_id:
        return rows
    new_rows = []
    for (x0, x1) in rects:
        start_ms, end_ms = (x0, x1) if x0 <= x1 else (x1, x0)
        new_rows.append({"start_ms": round(float(start_ms), 6),
                         "end_ms": round(float(end_ms), 6),
                         "trace_id": trace_id,
                         "label": "unknown"})
    return rows + new_rows

@app.callback(
    Output("labels-table", "data", allow_duplicate=True),
    Input("add-selection-long", "n_clicks"),
    State("labels-table", "data"),
    State("rects-store-long", "data"),
    prevent_initial_call=True
)
def add_sel_long(n_clicks, rows, rects):
    if not n_clicks:
        return no_update
    if not rects:
        return rows
    new_rows = []
    for (x0, x1) in rects:
        start_ms, end_ms = (x0, x1) if x0 <= x1 else (x1, x0)
        new_rows.append({"start_ms": round(float(start_ms), 6),
                         "end_ms": round(float(end_ms), 6),
                         "trace_id": "long_trace",
                         "label": "unknown"})
    return rows + new_rows


@app.callback(
    Output("trace-graph", "figure", allow_duplicate=True),
    Output("rects-store-sweep", "data", allow_duplicate=True),
    Input("erase-selection-sweep", "n_clicks"),
    State("trace-graph", "figure"),
    prevent_initial_call=True
)
def erase_shapes_sweep(n_clicks, fig):
    if not n_clicks: return no_update, no_update
    fig["layout"]["shapes"] = []
    return fig, []

@app.callback(
    Output("long-graph", "figure", allow_duplicate=True),
    Output("rects-store-long", "data", allow_duplicate=True),
    Input("erase-selection-long", "n_clicks"),
    State("long-graph", "figure"),
    prevent_initial_call=True
)
def erase_shapes_long(n_clicks, fig):
    if not n_clicks: return no_update, no_update
    fig["layout"]["shapes"] = []
    return fig, []


@app.callback(
    Output("labels-table", "data", allow_duplicate=True),
    Output("save-status", "children"),
    Input("save-labels", "n_clicks"),
    State("labels-table", "data"),
    prevent_initial_call=True
)
def save_labels(n_clicks, rows):
    if not n_clicks or not rows:
        return no_update, ""
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patient_info": PATIENT,
        "device_info": DEVICE,
        "labels": [],
    }
    for r in rows:
        try:
            start_ms = float(r["start_ms"])
            end_ms = float(r["end_ms"])
            if end_ms < start_ms:
                start_ms, end_ms = end_ms, start_ms
            trace_id = r["trace_id"]
            label = r.get("label", "unknown")
            if trace_id not in TRACES:
                continue
            x, y, _ = get_trace_xy(trace_id)
            mask = (x >= start_ms) & (x <= end_ms)
            segment = [{"time_ms": float(t), "voltage_uv": float(v)} for t, v in zip(x[mask], y[mask])]
            payload["labels"].append({
                "trace_id": trace_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "label": label,
                "num_points": int(mask.sum()),
                "data": segment,
            })
        except Exception:
            continue
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = LABELED_DIR / f"labels_{ts}.json"
    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return [], f"Saved to {out_path.name} (UTF-16)."

@app.callback(
    Output("labels-table", "data", allow_duplicate=True),
    Input("clear-labels", "n_clicks"),
    prevent_initial_call=True
)

def clear_labels(n_clicks):
    if not n_clicks:
        return no_update
    return []

if __name__ == "__main__":
    app.run(debug=True)
