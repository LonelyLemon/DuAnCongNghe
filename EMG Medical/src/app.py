import time
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update

from src.processing.loader import load_natus_txt
from src.processing.stats import calculate_clinical_stats

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data_raw"
LABELED_DIR = BASE_DIR / "labeled_data"
LABELED_DIR.mkdir(exist_ok=True)

raw_file = DATA_RAW_DIR / "Le Thi Bop.txt"

if raw_file.exists():
    emg_data = load_natus_txt(raw_file)
else:
    print(f"Không tìm thấy file {raw_file}")
    emg_data = {}

full_stream = emg_data.get("full_data_stream", [])
dt_ms = emg_data.get("full_sequence", {}).get("dt_ms", 0.02)
boundaries = emg_data.get("full_sequence", {}).get("boundaries", [])
patient = emg_data.get("patient_info", {})

if full_stream:
    full_voltage = np.array(full_stream, dtype=float)
    full_time = np.arange(len(full_voltage), dtype=float) * dt_ms
else:
    full_voltage, full_time = np.array([]), np.array([])

# --- APP LAYOUT ---
app = Dash(__name__)
app.title = "EMG Lab - Smart Analysis"

app.layout = html.Div([
    html.H2("EMG Lab - Hệ thống Phân tích & Gán nhãn Thông minh", 
            style={"textAlign": "center", "color": "#111827"}),

    # INFO PANEL
    html.Div([
        html.Div([
            html.H4(f"Bệnh nhân: {patient.get('first_name', 'N/A')} ({patient.get('patient_id', '')})"),
            html.P(f"Ngày khám: {patient.get('visit_date', '')}"),
            html.P(f"Bài đo: {patient.get('test_name', '')}")
        ], style={"flex": 1, "backgroundColor": "#eef2ff", "padding": "10px", "borderRadius": "8px"}),
        
        html.Div([
            html.H4("Kết quả phân tích vùng chọn (Real-time DSP)", style={"color": "#b91c1c"}),
            html.Div(id="stats-display", children="Hãy chọn một vùng trên đồ thị để phân tích...", 
                     style={"fontWeight": "bold", "fontSize": "16px"})
        ], style={"flex": 1, "backgroundColor": "#fef2f2", "padding": "10px", "borderRadius": "8px", "border": "1px solid #b91c1c"})
    ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

    # GRAPH
    html.Div([
        dcc.Graph(id="timeline-graph", style={"height": "60vh"}, 
                  config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToAdd": ["drawrect", "eraseshape"]})
    ], style={"boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1)"}),

    # CONTROLS
    html.Div([
        html.Button("➕ Gán nhãn vùng này", id="btn-add", style={"padding": "10px 20px", "background": "#2563eb", "color": "white", "border": "none", "borderRadius": "5px"}),
        html.Span(id="msg-log", style={"marginLeft": "15px", "color": "gray"})
    ], style={"marginTop": "15px"}),

    # LABEL TABLE
    html.H4("Danh sách nhãn"),
    dash_table.DataTable(
        id="table-labels",
        columns=[
            {"name": "Start(ms)", "id": "start"}, 
            {"name": "End(ms)", "id": "end"},
            {"name": "P2P(µV)", "id": "p2p"},
            {"name": "RMS(µV)", "id": "rms"},
            {"name": "Trace", "id": "trace"},
            {"name": "Label", "id": "label", "presentation": "dropdown"}
        ],
        data=[],
        editable=True,
        dropdown={"label": {"options": [{"label": "Bệnh lý", "value": "PATH"}, {"label": "Bình thường", "value": "NORM"}]}},
        style_header={"backgroundColor": "#f3f4f6", "fontWeight": "bold"}
    ),
    dcc.Store(id="current-selection-stats"),
    dcc.Store(id="relayout-data")
], style={"fontFamily": "Segoe UI, sans-serif", "maxWidth": "1400px", "margin": "0 auto", "padding": "20px"})

# --- CALLBACKS ---

@app.callback(
    Output("timeline-graph", "figure"),
    Input("timeline-graph", "id")
)
def init_graph(_):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=full_time, y=full_voltage, mode='lines', line=dict(color='#374151', width=1), name="EMG"))
    
    shapes = [dict(type="line", x0=b["start_ms"], x1=b["start_ms"], y0=0, y1=1, yref="paper", line=dict(color="rgba(0,0,0,0.1)")) for b in boundaries[:500]]
    
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True), title="Time (ms)"),
        yaxis=dict(title="Amplitude (µV)"),
        shapes=shapes, template="plotly_white", margin=dict(l=50, r=20, t=20, b=40)
    )
    return fig

@app.callback(
    Output("stats-display", "children"),
    Output("current-selection-stats", "data"),
    Output("relayout-data", "data"),
    Input("timeline-graph", "relayoutData"),
    prevent_initial_call=True
)
def analyze_selection(relayout):
    if not relayout: return no_update, no_update, no_update
    
    x0, x1 = None, None
    if "shapes" in relayout:
        x0, x1 = relayout["shapes"][-1]["x0"], relayout["shapes"][-1]["x1"]
    elif "xaxis.range[0]" in relayout:
        x0, x1 = relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]
        
    if x0 is None: return "Chưa chọn vùng cụ thể", {}, relayout

    s, e = sorted([float(x0), float(x1)])
    
    mask = (full_time >= s) & (full_time <= e)
    seg_v = full_voltage[mask]
    seg_t = full_time[mask]
    
    stats = calculate_clinical_stats(seg_t, seg_v)
    
    display_text = f"P2P: {stats['p2p_uv']} µV | RMS: {stats['rms_uv']} µV | Duration: {stats['duration_ms']} ms | Turns: {stats['turns']}"
    
    stats["start"] = s
    stats["end"] = e
    
    return display_text, stats, relayout

@app.callback(
    Output("table-labels", "data"),
    Output("msg-log", "children"),
    Input("btn-add", "n_clicks"),
    State("current-selection-stats", "data"),
    State("table-labels", "data"),
    prevent_initial_call=True
)
def add_label(n_clicks, stats, rows):
    if not stats: return no_update, "Chưa có thông số để thêm."
    
    mid = (stats["start"] + stats["end"]) / 2
    trace_id = "N/A"
    for b in boundaries:
        if b["start_ms"] <= mid < b["end_ms"]:
            trace_id = b["trace_id"]; break

    new_row = {
        "start": round(stats["start"], 1),
        "end": round(stats["end"], 1),
        "p2p": stats["p2p_uv"],
        "rms": stats["rms_uv"],
        "trace": trace_id,
        "label": "PATH"
    }
    
    return (rows or []) + [new_row], f"Đã lưu vùng {trace_id}"

if __name__ == "__main__":
    app.run(debug=True)