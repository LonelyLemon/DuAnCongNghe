import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import base64
import io
import os
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, callback_context

# --- IMPORT MODULES ---
from src.database.db_manager import get_all_recordings, get_recording_by_id, save_label_to_db
from src.processing.loader import load_natus_txt, get_data_slice, get_downsampled_data, parse_natus_content
from src.processing.stats import calculate_clinical_stats
from src.processing.filters import apply_notch_filter, apply_bandpass_filter
from src.database.db_manager import add_patient_if_not_exists, add_recording, get_connection
from src.reporting.generator import generate_pdf_buffer

# --- CONFIG ---
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "EMG Lab - Smart Analysis"

# Layout Container
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        html.H2("EMG LAB", style={"margin": 0, "color": "white"}),
        html.Div([
            dcc.Link("Trang chủ", href="/", style={"color": "white", "marginRight": "20px", "textDecoration": "none"}),
            html.Span("v1.2 (Interactive Filters)", style={"color": "#9ca3af", "fontSize": "14px"})
        ], style={"display": "flex", "alignItems": "center"})
    ], style={"backgroundColor": "#111827", "padding": "15px 30px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
    html.Div(id="page-content", style={"padding": "20px", "maxWidth": "1400px", "margin": "0 auto"})
])

# --- TRANG CHỦ ---
def layout_home():
    recordings = get_all_recordings()
    table_data = [{"id": r["id"], "date": r["visit_date"], "patient": f"{r['full_name']} ({r['patient_code']})", "test": r["test_name"], "action": "Xem chi tiết"} for r in recordings]
    return html.Div([
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Kéo thả hoặc ',
                    html.A('Chọn File EMG (.txt)', style={'color': '#2563eb', 'fontWeight': 'bold', 'cursor': 'pointer'})
                ]),
                style={
                    'width': '100%', 'height': '80px', 'lineHeight': '80px',
                    'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center', 'margin': '20px 0', 'backgroundColor': '#f9fafb',
                    'borderColor': '#d1d5db', 'color': '#4b5563'
                },
                multiple=True
            ),
            html.Div(id='upload-status', style={'textAlign': 'center', 'marginBottom': '20px', 'fontWeight': 'bold'})
        ]),
        html.H3("Danh sách bản ghi EMG", style={"borderBottom": "2px solid #e5e7eb", "paddingBottom": "10px"}),

        dcc.Input(id="search-input", type="text", placeholder="🔍 Tìm theo tên hoặc mã bệnh nhân...", 
                  style={"padding": "10px", "width": "300px", "marginBottom": "15px", "borderRadius": "5px", "border": "1px solid #ccc"}),
        
        dash_table.DataTable(
            id="table-recordings",
            columns=[{"name": "ID", "id": "id"}, {"name": "Ngày khám", "id": "date"}, {"name": "Bệnh nhân", "id": "patient"}, {"name": "Bài đo", "id": "test"}, {"name": "Thao tác", "id": "action", "presentation": "markdown"}],
            data=table_data,
            style_cell={"padding": "12px", "textAlign": "left", "fontFamily": "Segoe UI"},
            style_header={"backgroundColor": "#f3f4f6", "fontWeight": "bold"},
            style_data_conditional=[{'if': {'column_id': 'action'}, 'color': 'blue', 'cursor': 'pointer'}],
            row_selectable="single"
        ),
        html.Div(id="hidden-redirect")
    ])

# --- TRANG PHÂN TÍCH ---
def layout_analysis(rec_id):
    rec = get_recording_by_id(rec_id)
    if not rec: return html.Div("❌ Không tìm thấy bản ghi.", style={"color": "red"})
    
    file_path = Path(rec["file_path"])
    if not file_path.exists():
        fallback = Path(__file__).parent.parent / "data_raw" / file_path.name
        if fallback.exists(): file_path = fallback
        else: return html.Div("❌ File gốc không tồn tại.", style={"color": "red"})

    # Load dữ liệu nén
    ds_time, ds_volt, boundaries = get_downsampled_data(file_path, max_points=5000)
    dt_ms = boundaries[0]["end_ms"] / len(ds_time) if len(ds_time) > 0 else 0.02 # Ước lượng dt

    # Initial View
    initial_end = boundaries[0]["end_ms"] if boundaries else 100.0

    return html.Div([
        dcc.Store(id="current-rec-id", data=rec_id),
        dcc.Store(id="current-file-path", data=str(file_path)),
        dcc.Store(id="current-boundaries", data=boundaries), 
        dcc.Store(id="current-dt-ms", data=dt_ms), # Lưu dt để dùng cho bộ lọc
        
        # Info Bar
        html.Div([
            html.Div([
                html.Strong(f"{rec['visit_date']} | {rec['test_name']}"),
                html.Br(),
                html.Span(f"Bệnh nhân: {rec.get('full_name', 'Unknown')}", style={"color": "gray"})
            ]),
            
            # [MỚI] KHU VỰC BỘ LỌC
            html.Div([
                html.Span("Bộ lọc tín hiệu (DSP):", style={"fontWeight": "bold", "marginRight": "10px", "color": "#4b5563"}),
                dcc.Checklist(
                    id="dsp-filters-checklist",
                    options=[
                        {'label': ' Lọc nhiễu nguồn (Notch 50Hz)', 'value': 'NOTCH'},
                        {'label': ' Lọc dải (Bandpass 20-500Hz)', 'value': 'BANDPASS'},
                    ],
                    value=[], # Mặc định không lọc
                    inline=True,
                    inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                )
            ], style={"display": "flex", "alignItems": "center", "backgroundColor": "#f3f4f6", "padding": "5px 15px", "borderRadius": "20px", "border": "1px solid #d1d5db", "marginLeft": "20px"}),

            html.Div(id="dsp-stats-display", style={"color": "#b91c1c", "fontWeight": "bold", "marginLeft": "auto", "alignSelf": "center"})
        ], style={"backgroundColor": "#e0f2fe", "padding": "15px", "borderRadius": "8px", "display": "flex", "marginBottom": "15px", "alignItems": "center"}),
        
        # Graph
        dcc.Loading(
            id="loading-graph",
            type="default",
            children=dcc.Graph(
                id="analysis-graph", 
                style={"height": "65vh"},
                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                # Lưu ý: Figure ban đầu chưa lọc (để nhanh). Khi user tích lọc, callback sẽ chạy lại.
                figure=create_initial_figure(ds_time, ds_volt, boundaries, initial_end)
            )
        ),
        
        # Controls
        html.Div([
            html.Button("➕ Lưu nhãn vùng chọn", id="btn-save-label", style={"padding": "10px 20px", "background": "#16a34a", "color": "white", "border": "none", "borderRadius": "5px"}),
            html.Button("📄 Xuất Báo Cáo PDF", id="btn-export-pdf", style={"padding": "10px 20px", "background": "#ea580c", "color": "white", "border": "none", "borderRadius": "5px"}),
            dcc.Download(id="download-pdf"),
            dcc.Dropdown(
                id="label-type-dd",
                options=[{"label": "Bệnh lý", "value": "PATH"}, {"label": "Bình thường", "value": "NORM"}, {"label": "Nhiễu", "value": "ARTIFACT"}],
                value="PATH",
                style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}
            ),
            html.Span(id="save-msg", style={"marginLeft": "15px", "color": "gray"})
        ], style={"marginTop": "15px", "padding": "10px", "backgroundColor": "#f9fafb"}),
        
        dcc.Store(id="temp-stats-store")
    ])

def create_initial_figure(time, voltage, bounds, initial_end):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=time, y=voltage, mode='lines', line=dict(color='#6b7280', width=1), name="Signal", hoverinfo='skip'))
    
    limit_shapes = bounds[:100] if len(bounds) > 100 else bounds
    shapes = [dict(type="line", x0=b["start_ms"], x1=b["start_ms"], y0=0, y1=1, yref="paper", line=dict(color="rgba(0,0,0,0.1)")) for b in limit_shapes]
    
    fig.update_layout(
        margin=dict(l=50, r=20, t=20, b=40), template="plotly_white",
        xaxis=dict(title="Time (ms)", rangeslider=dict(visible=True), range=[0, initial_end]),
        yaxis=dict(title="Amplitude (µV)", fixedrange=False),
        shapes=shapes, dragmode="pan"
    )
    return fig

# --- CALLBACKS ---

# 1. Callback "DYNAMIC RELOAD & FILTER"
@app.callback(
    Output("analysis-graph", "figure"),
    Input("analysis-graph", "relayoutData"),
    Input("dsp-filters-checklist", "value"), # Trigger khi user đổi bộ lọc
    State("analysis-graph", "figure"),
    State("current-file-path", "data"),
    State("current-dt-ms", "data"),
    prevent_initial_call=True
)
def update_graph_content(relayout, active_filters, fig, file_path, dt_ms):
    if not file_path: return no_update
    
    try:
        if fig and "layout" in fig and "xaxis" in fig["layout"]:
            xaxis = fig["layout"]["xaxis"]
            if "rangeslider" in xaxis and "yaxis" in xaxis["rangeslider"]:
                xaxis["rangeslider"]["yaxis"].pop("_template", None)
    except Exception:
        pass

    ctx_msg = callback_context.triggered[0]['prop_id']
    
    # Xác định Range hiện tại
    if fig and "layout" in fig and "xaxis" in fig["layout"]:
        current_range = fig["layout"]["xaxis"].get("range", [0, 100])
        x0, x1 = current_range[0], current_range[1]
    else:
        x0, x1 = 0, 100

    # Nếu trigger là do zoom/pan, lấy range mới
    if relayout:
        if "xaxis.range[0]" in relayout:
            x0, x1 = relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]
        elif "xaxis.range" in relayout:
            x0, x1 = relayout["xaxis.range"][0], relayout["xaxis.range"][1]
            
    start_ms, end_ms = float(x0), float(x1)
    duration = end_ms - start_ms
    
    # Logic Load Data
    new_fig = go.Figure(fig)
    THRESHOLD_MS = 5000.0
    
    if duration < THRESHOLD_MS:
        # Load High-Res Slice
        # Lấy dư ra 50ms mỗi bên để lọc không bị méo ở biên (Boundary effect)
        pad = 50.0 
        hr_time, hr_vals = get_data_slice(Path(file_path), start_ms - pad, end_ms + pad)
        
        # [QUAN TRỌNG] ÁP DỤNG BỘ LỌC
        processed_vals = hr_vals.copy()
        
        # Lấy dt chính xác hơn từ slice (nếu có)
        current_slice_dt = (hr_time[1] - hr_time[0]) if len(hr_time) > 1 else (dt_ms or 0.02)

        if active_filters:
            if "NOTCH" in active_filters:
                processed_vals = apply_notch_filter(processed_vals, current_slice_dt)
            if "BANDPASS" in active_filters:
                processed_vals = apply_bandpass_filter(processed_vals, current_slice_dt)

        # Cắt bỏ phần dư (padding) sau khi lọc xong để hiển thị đẹp
        mask = (hr_time >= start_ms) & (hr_time <= end_ms)
        plot_time = hr_time[mask]
        plot_vals = processed_vals[mask]
        
        # Update Trace 0
        if len(new_fig.data) > 0:
            new_fig.data[0].x = plot_time
            new_fig.data[0].y = plot_vals
            new_fig.data[0].line.color = "#111827" if not active_filters else "#059669" # Đổi màu xanh nếu đã lọc
            new_fig.data[0].name = "Filtered (High-Res)" if active_filters else "Raw (High-Res)"
            
    else:
        # Nếu đang ở chế độ Overview (Zoom out xa)
        # Để đơn giản, ta KHÔNG lọc overview (vì nó đã bị downsample, lọc sai)
        # Chỉ đổi màu về xám để user biết đây là overview
        if len(new_fig.data) > 0:
            new_fig.data[0].line.color = "#6b7280"
            new_fig.data[0].name = "Overview (No Filter)"

    new_fig.update_layout(xaxis=dict(range=[start_ms, end_ms]))
    return new_fig

# 2. Stats Calculator (Cần cập nhật để tính trên dữ liệu ĐÃ LỌC)
@app.callback(
    Output("dsp-stats-display", "children"),
    Output("temp-stats-store", "data"),
    Input("analysis-graph", "relayoutData"),
    State("current-file-path", "data"),
    State("dsp-filters-checklist", "value"), # Lấy trạng thái bộ lọc
    prevent_initial_call=True
)
def update_stats_display(relayout, file_path, active_filters):
    if not relayout: return no_update, no_update
    
    if "shapes" in relayout:
        x0, x1 = relayout["shapes"][-1]["x0"], relayout["shapes"][-1]["x1"]
        s, e = sorted([float(x0), float(x1)])
        
        # Load Raw Data
        t_arr, v_arr = get_data_slice(Path(file_path), s, e)
        dt = (t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 0.02
        
        # Áp dụng bộ lọc y hệt như trên đồ thị
        processed_vals = v_arr.copy()
        if active_filters:
            if "NOTCH" in active_filters:
                processed_vals = apply_notch_filter(processed_vals, dt)
            if "BANDPASS" in active_filters:
                processed_vals = apply_bandpass_filter(processed_vals, dt)
        
        # Tính toán trên dữ liệu đã lọc
        stats = calculate_clinical_stats(t_arr, processed_vals)
        
        filter_status = "(Đã lọc)" if active_filters else "(Raw)"
        return f"P2P: {stats.get('p2p_uv', 0)}µV | RMS: {stats.get('rms_uv', 0)}µV {filter_status}", {"start": s, "end": e, "stats": stats}
        
    return no_update, no_update

# 3. Routing & Save Label (Giữ nguyên logic cũ)
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/": return layout_home()
    elif pathname.startswith("/analysis/"):
        try: return layout_analysis(int(pathname.split("/")[-1]))
        except: return "Lỗi ID"
    return "404"

@app.callback(Output("url", "pathname"), Input("table-recordings", "selected_rows"), State("table-recordings", "data"), prevent_initial_call=True)
def go_to_analysis(selected_rows, rows):
    if selected_rows: return f"/analysis/{rows[selected_rows[0]]['id']}"
    return no_update

@app.callback(
    Output("save-msg", "children"),
    Input("btn-save-label", "n_clicks"),
    State("current-rec-id", "data"),
    State("temp-stats-store", "data"),
    State("label-type-dd", "value"),
    State("current-boundaries", "data"),
    prevent_initial_call=True
)
def save_label_db(n_clicks, rec_id, stats_data, label_type, boundaries):
    if not stats_data: return "Chưa chọn vùng!"
    s, e = stats_data["start"], stats_data["end"]
    vals = stats_data.get("stats", {})
    mid = (s+e)/2
    trace_id = "Unknown"
    for b in boundaries:
        if b["start_ms"] <= mid < b["end_ms"]: trace_id = b["trace_id"]; break
    save_label_to_db(rec_id, s, e, trace_id, label_type, vals.get("p2p_uv",0), vals.get("rms_uv",0))
    return f"✅ Đã lưu {label_type} ({trace_id})"

@app.callback(
    Output('upload-status', 'children'),
    Output('table-recordings', 'data'), # Cập nhật lại bảng sau khi upload
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('table-recordings', 'data'), # Lấy dữ liệu bảng hiện tại để update hoặc giữ nguyên
    prevent_initial_call=True
)
def update_output(list_of_contents, list_of_names, current_rows):
    if list_of_contents is None:
        return no_update, no_update

    messages = []
    
    # Thư mục lưu file vật lý (tạo nếu chưa có)
    # Ta vẫn CẦN lưu file ra ổ cứng để hàm 'get_data_slice' (lazy load) hoạt động về sau
    save_dir = Path(__file__).parent.parent / "data_raw" / "imported"
    save_dir.mkdir(parents=True, exist_ok=True)

    for content, name in zip(list_of_contents, list_of_names):
        if not name.lower().endswith('.txt'):
            messages.append(html.P(f"❌ {name}: Không phải file .txt", style={'color': 'red'}))
            continue
            
        try:
            # 1. Giải mã nội dung (Base64 -> String)
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            
            # Lưu ý: File gốc là UTF-16, nên ta decode UTF-16
            try:
                text_content = decoded.decode('utf-16')
            except UnicodeDecodeError:
                # Fallback nếu không phải utf-16
                text_content = decoded.decode('utf-8', errors='ignore')

            # 2. Parse để lấy thông tin Metadata
            data = parse_natus_content(text_content)
            p_info = data.get('patient_info', {})
            
            if not p_info.get('patient_id'):
                messages.append(html.P(f"⚠️ {name}: Không tìm thấy thông tin bệnh nhân hợp lệ.", style={'color': 'orange'}))
                # Vẫn có thể lưu nhưng cảnh báo
            
            # 3. Lưu file vật lý xuống ổ cứng
            # Để tránh trùng tên, có thể thêm timestamp vào tên file
            import time
            safe_name = f"{int(time.time())}_{name}"
            file_path = save_dir / safe_name
            
            # Ghi file (encoding utf-16 để giữ nguyên gốc)
            with open(file_path, "w", encoding="utf-16") as f:
                f.write(text_content)
                
            # 4. Ghi vào Database
            p_id = add_patient_if_not_exists(
                patient_code=p_info.get('patient_id', 'UNKNOWN'),
                full_name=p_info.get('first_name', 'Unknown Import')
            )
            
            add_recording(
                patient_id=p_id,
                visit_date=p_info.get('visit_date'),
                test_name=p_info.get('test_name'),
                file_path=str(file_path),
                duration_ms=0
            )
            
            messages.append(html.P(f"✅ {name}: Import thành công!", style={'color': 'green'}))
            
        except Exception as e:
            messages.append(html.P(f"❌ {name}: Lỗi xử lý ({str(e)})", style={'color': 'red'}))

    # Load lại danh sách bản ghi mới nhất từ DB
    new_recordings = get_all_recordings()
    new_table_data = [{"id": r["id"], "date": r["visit_date"], "patient": f"{r['full_name']} ({r['patient_code']})", "test": r["test_name"], "action": "Xem chi tiết"} for r in new_recordings]

    return messages, new_table_data

@app.callback(
    Output("table-recordings", "data", allow_duplicate=True),
    Input("search-input", "value"),
    prevent_initial_call=True
)
def filter_table(search_term):
    # Lấy toàn bộ data mới nhất
    all_recs = get_all_recordings()
    base_data = [{"id": r["id"], "date": r["visit_date"], "patient": f"{r['full_name']} ({r['patient_code']})", "test": r["test_name"], "action": "Xem chi tiết"} for r in all_recs]
    
    if not search_term:
        return base_data
    
    # Lọc (Case insensitive)
    term = search_term.lower()
    filtered = [
        row for row in base_data 
        if term in row['patient'].lower() or term in str(row['date']).lower()
    ]
    return filtered

@app.callback(
    Output("download-pdf", "data"),
    Input("btn-export-pdf", "n_clicks"),
    State("current-rec-id", "data"),
    State("current-file-path", "data"),
    prevent_initial_call=True
)
def export_pdf_report(n_clicks, rec_id, file_path_str):
    if not rec_id: return no_update
    
    # 1. Lấy thông tin bản ghi & Bệnh nhân từ DB
    rec = get_recording_by_id(rec_id) # Hàm này bạn đã có trong db_manager nhưng cần chỉnh để join lấy tên BN
    # (Để đơn giản, ta query thủ công ở đây hoặc cập nhật db_manager sau. Tạm thời giả định rec có đủ field)
    
    # Để chắc chắn, ta query lại labels của rec_id này
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM labels WHERE recording_id = ?", (rec_id,))
    labels_rows = cur.fetchall()
    conn.close()
    
    if not labels_rows:
        return no_update # Hoặc hiện thông báo "Chưa có nhãn nào để in"
    
    # 2. Chuẩn bị dữ liệu vẽ (Re-fetch raw data)
    file_path = Path(file_path_str)
    labels_data = []
    
    for row in labels_rows:
        s, e = row['start_ms'], row['end_ms']
        # Load lại dữ liệu raw đoạn này để vẽ cho nét
        t_arr, v_arr = get_data_slice(file_path, s, e)
        
        labels_data.append({
            "trace_id": row['trace_id'],
            "start_ms": s, "end_ms": e,
            "label_type": row['label_type'],
            "stats": {"p2p_uv": row['p2p_uv'], "rms_uv": row['rms_uv'], "duration_ms": round(e-s, 1)},
            "time_arr": t_arr,
            "volt_arr": v_arr
        })
    
    # 3. Tạo PDF
    # Mock patient info nếu rec chưa join (Tuỳ db_manager của bạn)
    # Tốt nhất là dùng rec đã lấy ở layout_analysis
    patient_info = {
        "full_name": rec.get('full_name', 'Unknown'),
        "patient_code": rec.get('patient_code', '---')
    }
    
    pdf_buf = generate_pdf_buffer(patient_info, rec, labels_data)
    
    if pdf_buf:
        filename = f"Report_{rec.get('patient_code', 'PAT')}_{rec.get('visit_date', '').replace('/','-')}.pdf"
        return dcc.send_bytes(pdf_buf.getvalue(), filename)
    
    return no_update

if __name__ == "__main__":
    app.run(debug=True)