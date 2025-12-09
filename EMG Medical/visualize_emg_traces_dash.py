import json
import time
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List, Optional

from dash import (
    Dash, 
    dcc, 
    html, 
    Input, 
    Output, 
    State,
    dash_table,
    no_update, 
    ctx
)

# ---------- CẤU HÌNH ĐƯỜNG DẪN ----------
BASE_DIR = Path(__file__).resolve().parent
EMG_JSON = BASE_DIR / "data_processed" / "emg_data.json"
LABELED_DIR = BASE_DIR / "labeled_data"
LABELED_DIR.mkdir(exist_ok=True)

# ---------- LOAD DỮ LIỆU ĐÃ XỬ LÝ ----------
# Lưu ý: File json phải được tạo từ main.py mới nhất (đã có logic nối chuỗi)
try:
    with open(EMG_JSON, "r", encoding="utf-16") as f:
        emg = json.load(f)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {EMG_JSON}. Hãy chạy main.py trước.")
    emg = {}

DEVICE = emg.get("device_info", {})
PATIENT = emg.get("patient_info", {}) or {}

# Lấy dữ liệu chuỗi nối tiếp (Full Sequence)
FULL_SEQ = emg.get("full_sequence", {})
FULL_STREAM = emg.get("full_data_stream", {}).get("voltage_uv", [])
BOUNDARIES = FULL_SEQ.get("boundaries", []) # Danh sách mốc thời gian phân chia các Trace

# Tái tạo mảng thời gian và điện thế
# Nếu không có full_sequence (do chạy main.py cũ), fallback về rỗng
dt_ms = FULL_SEQ.get("dt_ms", 0.02)
total_points = len(FULL_STREAM)

if total_points > 0:
    full_time = np.arange(total_points, dtype=float) * dt_ms
    full_voltage = np.array(FULL_STREAM, dtype=float)
else:
    full_time = np.array([])
    full_voltage = np.array([])
    print("⚠️ Cảnh báo: Không tìm thấy dữ liệu 'full_data_stream'. Hãy đảm bảo bạn đã cập nhật và chạy lại main.py.")

# ---------- GIAO DIỆN DASHBOARD ----------

app = Dash(__name__)
app.title = "EMG Timeline Visualizer"

# Bảng thông tin bệnh nhân
patient_data = [{"Field": k, "Value": v} for k, v in PATIENT.items() if v]
patient_table = dash_table.DataTable(
    id="patient-table",
    columns=[{"name": "Thông tin", "id": "Field"}, {"name": "Chi tiết", "id": "Value"}],
    data=patient_data,
    style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14, "textAlign": "left"},
    style_header={"fontWeight": "bold", "backgroundColor": "#f3f4f6"},
    style_table={"maxHeight": "250px", "overflowY": "auto", "border": "1px solid #e5e7eb"},
)

# Bảng quản lý nhãn (Labels)
labels_table = dash_table.DataTable(
    id="labels-table",
    columns=[
        {"name": "Start (ms)", "id": "start_ms", "type": "numeric"},
        {"name": "End (ms)", "id": "end_ms", "type": "numeric"},
        {"name": "Trace Gốc", "id": "trace_id", "type": "text"},
        {"name": "Nhãn bệnh", "id": "label", "presentation": "dropdown"},
    ],
    data=[],
    editable=True,
    row_deletable=True,
    dropdown={
        "label": {
            "options": [
                {"label": "Bệnh A", "value": "Bệnh A"},
                {"label": "Bệnh B", "value": "Bệnh B"},
                {"label": "Nhiễu (Artifact)", "value": "Artifact"},
                {"label": "Unknown", "value": "Unknown"},
            ]
        }
    },
    style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": 14, "textAlign": "left"},
    style_header={"fontWeight": "bold", "backgroundColor": "#f3f4f6"},
    style_table={"maxHeight": "250px", "overflowY": "auto", "border": "1px solid #e5e7eb"},
)

# Layout chính
app.layout = html.Div(
    style={"fontFamily": "system-ui, Arial", "padding": "20px", "maxWidth": "1600px", "margin": "0 auto", "backgroundColor": "#f9fafb", "minHeight": "100vh"},
    children=[
        html.H2("Phân tích tín hiệu EMG (Chế độ Timeline)", style={"color": "#111827", "marginBottom": "20px"}),

        # Khu vực thông tin & Bảng nhãn (Layout 2 cột)
        html.Div([
            html.Div([
                html.H4("Hồ sơ bệnh nhân", style={"marginTop": 0}),
                patient_table
            ], style={"flex": "0 0 350px"}), # Cố định chiều rộng cột trái

            html.Div([
                html.Div([
                    html.H4("Danh sách nhãn đã gán", style={"marginTop": 0, "display": "inline-block"}),
                    html.Div([
                        html.Button("Lưu nhãn ra file", id="btn-save-labels", n_clicks=0, 
                                    style={"backgroundColor": "#059669", "color": "white", "border": "none", "padding": "6px 12px", "borderRadius": "4px", "cursor": "pointer", "fontSize": "13px"}),
                        html.Span(id="save-status", style={"marginLeft": "10px", "fontSize": "13px", "color": "#059669"})
                    ], style={"float": "right"})
                ]),
                labels_table
            ], style={"flex": "1"}), # Cột phải co giãn
        ], style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        # Khu vực Đồ thị chính (Timeline)
        html.Div([
            dcc.Graph(
                id="timeline-graph",
                style={"height": "65vh"},
                # config bao gồm các công cụ vẽ
                config={
                    "scrollZoom": True, 
                    "displaylogo": False, 
                    "modeBarButtonsToAdd": ["drawrect", "eraseshape"]
                }
            )
        ], style={"backgroundColor": "white", "padding": "10px", "borderRadius": "8px", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"}),

        # Khu vực điều khiển gán nhãn
        html.Div([
            html.Div([
                html.Strong("Hướng dẫn: "),
                html.Span("Dùng công cụ 'Box Select' hoặc 'Draw Rectangle' trên thanh công cụ đồ thị để khoanh vùng bất thường. Sau đó bấm nút bên dưới.")
            ], style={"marginBottom": "10px", "color": "#4b5563"}),

            html.Button("➕ Thêm vùng đang chọn vào bảng nhãn", id="btn-add-label", n_clicks=0, 
                        style={"padding": "10px 20px", "backgroundColor": "#2563eb", "color": "white", "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
            html.Button("Xóa các vùng vẽ", id="btn-clear-shapes", n_clicks=0,
                        style={"padding": "10px 20px", "backgroundColor": "#9ca3af", "color": "white", "border": "none", "borderRadius": "6px", "cursor": "pointer", "marginLeft": "10px"}),
            html.Span(id="selection-msg", style={"marginLeft": "15px", "color": "#6b7280", "fontStyle": "italic"})
        ], style={"marginTop": "15px"}),
        
        # Stores để lưu trạng thái
        dcc.Store(id="relayout-data-store"),
    ]
)


# ---------- CALLBACKS ----------

# 1. Khởi tạo biểu đồ Timeline (Chạy 1 lần duy nhất khi load)
@app.callback(
    Output("timeline-graph", "figure"),
    Input("timeline-graph", "id")
)
def init_graph(_):
    fig = go.Figure()
    
    # Sử dụng WebGL (Scattergl) để vẽ mượt mà số lượng điểm lớn
    fig.add_trace(go.Scattergl(
        x=full_time,
        y=full_voltage,
        mode='lines',
        name='Tín hiệu EMG',
        line=dict(color='#1f2937', width=1),
        hoverinfo='x+y'
    ))

    # Tạo các vạch kẻ mờ phân chia các Trace gốc (dựa trên BOUNDARIES)
    # Chỉ vẽ nếu số lượng trace < 500 để tránh lag trình duyệt
    shapes = []
    if len(BOUNDARIES) > 0:
        for b in BOUNDARIES:
            shapes.append(dict(
                type="line",
                x0=b["start_ms"], x1=b["start_ms"],
                y0=0, y1=1, yref="paper", # Vẽ hết chiều cao đồ thị
                line=dict(color="rgba(200, 200, 200, 0.5)", width=1, dash="dot")
            ))
    
    fig.update_layout(
        title=dict(text="Biểu đồ tín hiệu toàn trình (Timeline)", font=dict(size=18)),
        xaxis=dict(
            title="Thời gian (ms)",
            rangeslider=dict(visible=True), # ĐÂY LÀ TÍNH NĂNG SLIDING WINDOW
            gridcolor="#f3f4f6"
        ),
        yaxis=dict(
            title="Biên độ (µV)",
            fixedrange=False, # Cho phép zoom trục Y
            gridcolor="#f3f4f6"
        ),
        shapes=shapes,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=40),
        dragmode="pan", # Mặc định là chế độ kéo để di chuyển
        hovermode="x unified"
    )
    return fig


# 2. Lưu trạng thái relayout (khi user vẽ hình hoặc zoom)
@app.callback(
    Output("relayout-data-store", "data"),
    Input("timeline-graph", "relayoutData"),
    prevent_initial_call=True
)
def store_relayout(relayout):
    return relayout


# 3. Xử lý nút "Thêm vùng chọn vào bảng"
@app.callback(
    Output("labels-table", "data"),
    Output("selection-msg", "children"),
    Input("btn-add-label", "n_clicks"),
    State("relayout-data-store", "data"),
    State("labels-table", "data"),
    prevent_initial_call=True
)
def add_label_from_selection(n_clicks, relayout, current_rows):
    if not relayout:
        return no_update, "Chưa có vùng nào được chọn."
    
    # Tìm tọa độ hình chữ nhật (User vẽ) hoặc vùng zoom (User zoom)
    x0, x1 = None, None
    
    # Trường hợp 1: Vẽ bằng công cụ Draw Rect/Box Select (Ưu tiên)
    if "shapes" in relayout:
        # Lấy hình vẽ cuối cùng
        last_shape = relayout["shapes"][-1]
        x0 = last_shape.get("x0")
        x1 = last_shape.get("x1")
    
    # Trường hợp 2: Zoom (Lấy toàn bộ vùng đang hiển thị)
    elif "xaxis.range[0]" in relayout:
        x0 = relayout["xaxis.range[0]"]
        x1 = relayout["xaxis.range[1]"]

    if x0 is None or x1 is None:
        return no_update, "Hãy dùng công cụ vẽ hình chữ nhật (Draw Rect) để chọn vùng chính xác."

    # Sắp xếp start/end
    start_ms, end_ms = sorted([float(x0), float(x1)])
    
    # Tự động xác định vùng này thuộc Trace gốc nào
    ref_trace = "N/A"
    # Tìm trace mà điểm giữa của vùng chọn rơi vào
    mid_point = (start_ms + end_ms) / 2
    for b in BOUNDARIES:
        if b["start_ms"] <= mid_point < b["end_ms"]:
            ref_trace = b["trace_id"]
            break
            
    new_row = {
        "start_ms": round(start_ms, 2),
        "end_ms": round(end_ms, 2),
        "trace_id": ref_trace,
        "label": "Unknown" # Mặc định
    }
    
    if current_rows is None:
        current_rows = []
        
    msg = f"✅ Đã thêm: {ref_trace} ({start_ms:.1f}ms - {end_ms:.1f}ms)"
    return current_rows + [new_row], msg


# 4. Xóa các hình vẽ trên đồ thị (Clear shapes)
@app.callback(
    Output("timeline-graph", "figure", allow_duplicate=True),
    Input("btn-clear-shapes", "n_clicks"),
    State("timeline-graph", "figure"),
    prevent_initial_call=True
)
def clear_shapes(n_clicks, fig):
    if not fig:
        return no_update
    
    # Giữ lại các đường kẻ dọc (Vertical lines) là trace boundaries
    # Các đường này thường được thêm vào layout.shapes lúc init.
    # Logic ở đây đơn giản là reset lại shapes về ban đầu (chỉ chứa vạch kẻ dọc)
    
    # Tái tạo lại list shapes gốc (chỉ chứa vạch phân chia trace)
    base_shapes = []
    if len(BOUNDARIES) > 0:
        for b in BOUNDARIES:
            base_shapes.append(dict(
                type="line", x0=b["start_ms"], x1=b["start_ms"],
                y0=0, y1=1, yref="paper",
                line=dict(color="rgba(200, 200, 200, 0.5)", width=1, dash="dot")
            ))
            
    fig["layout"]["shapes"] = base_shapes
    return fig


# 5. Lưu Labels ra file JSON
@app.callback(
    Output("save-status", "children"),
    Input("btn-save-labels", "n_clicks"),
    State("labels-table", "data"),
    prevent_initial_call=True
)
def save_labels_to_disk(n_clicks, rows):
    if not rows:
        return "Bảng trống, không có gì để lưu."
        
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"labels_{ts}.json"
    out_path = LABELED_DIR / filename
    
    payload = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patient_info": PATIENT,
        "device_info": DEVICE,
        "labels": []
    }

    # Trích xuất dữ liệu sóng tương ứng với label
    # Lưu ý: Việc trích xuất data từ mảng lớn có thể tốn resource nếu quá nhiều label
    for r in rows:
        try:
            s = float(r["start_ms"])
            e = float(r["end_ms"])
            # Lọc dữ liệu trong khoảng s -> e
            # Dùng numpy mask để lấy nhanh
            mask = (full_time >= s) & (full_time <= e)
            segment_time = full_time[mask]
            segment_volt = full_voltage[mask]
            
            # Format lại data để lưu
            segment_data = [
                {"t": round(float(t), 3), "v": round(float(v), 3)} 
                for t, v in zip(segment_time, segment_volt)
            ]
            
            label_entry = {
                "start_ms": s,
                "end_ms": e,
                "trace_id": r.get("trace_id"),
                "label": r.get("label"),
                "data_points": len(segment_data),
                "data_segment": segment_data # Lưu kèm đoạn sóng
            }
            payload["labels"].append(label_entry)
        except Exception as ex:
            print(f"Lỗi khi xử lý dòng {r}: {ex}")
            continue

    with open(out_path, "w", encoding="utf-16") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        
    return f"Đã lưu thành công: {filename}"


if __name__ == "__main__":
    app.run(debug=True)