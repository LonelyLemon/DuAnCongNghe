import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update, callback_context, MATCH, ALL
import base64
import time

# IMPORT MODULES
from src.database.db_manager import (
    get_all_recordings, get_recording_by_id, 
    save_label_to_db, get_labels_by_recording, 
    delete_label_by_id, add_patient_if_not_exists, 
    add_recording, init_db,
    get_all_label_defs, add_label_def
)
from src.processing.loader import get_data_slice, get_downsampled_data, parse_natus_content
from src.processing.stats import calculate_clinical_stats
from src.processing.filters import apply_notch_filter, apply_bandpass_filter
from src.reporting.generator import generate_pdf_buffer
from src.utils import get_base_path

# --- CONFIG ---
init_db()

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "EMG Lab - Pro"

# Layout Container
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div([
        html.H2("EMG LAB PRO", style={"margin": 0, "color": "white"}),
        html.Div([
            dcc.Link("Trang chủ", href="/", style={"color": "white", "marginRight": "20px", "textDecoration": "none", "fontWeight": "bold"}),
            html.Span("v2.0 (Stable)", style={"color": "#9ca3af", "fontSize": "14px"})
        ], style={"display": "flex", "alignItems": "center"})
    ], style={"backgroundColor": "#111827", "padding": "15px 30px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
    html.Div(id="page-content", style={"padding": "20px", "maxWidth": "1600px", "margin": "0 auto"})
])

# --- TRANG CHỦ ---
def layout_home():
    recordings = get_all_recordings()
    table_data = []
    for r in recordings:
        table_data.append({
            "id": r["id"],
            "date": r["visit_date"],
            "patient": f"{r['full_name']} ({r['patient_code']})",
            "test": r["test_name"],
            "action": f"[Xem chi tiết](/analysis/{r['id']})"
        })

    return html.Div([
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Kéo thả file .txt vào đây hoặc ', html.A('Chọn File', style={'color': '#2563eb', 'fontWeight': 'bold'})]),
                style={'width': '100%', 'height': '80px', 'lineHeight': '80px', 'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px', 'textAlign': 'center', 'backgroundColor': '#f9fafb', 'color': '#4b5563'},
                multiple=True
            ),
            html.Div(id='upload-status', style={'textAlign': 'center', 'marginTop': '10px', 'fontWeight': 'bold'})
        ]),
        
        html.H3("Danh sách bản ghi", style={"marginTop": "20px"}),
        dcc.Input(id="search-input", type="text", placeholder="🔍 Tìm bệnh nhân...", style={"padding": "8px", "width": "300px", "marginBottom": "10px"}),
        
        dash_table.DataTable(
            id="table-recordings",
            columns=[
                {"name": "ID", "id": "id"}, 
                {"name": "Ngày khám", "id": "date"}, 
                {"name": "Bệnh nhân", "id": "patient"}, 
                {"name": "Bài đo", "id": "test"}, 
                {"name": "Thao tác", "id": "action", "presentation": "markdown"}
            ],
            data=table_data,
            style_cell={"padding": "10px", "textAlign": "left"},
            style_header={"backgroundColor": "#f3f4f6", "fontWeight": "bold"},
            page_size=10
        )
    ])

# --- TRANG PHÂN TÍCH ---
def layout_analysis(rec_id):
    rec = get_recording_by_id(rec_id)
    if not rec: return html.Div("❌ Không tìm thấy bản ghi.", style={"color": "red"})
    
    file_path = Path(rec["file_path"])
    if not file_path.exists():
        fallback = get_base_path() / "data_raw" / file_path.name
        if fallback.exists(): file_path = fallback
        else: return html.Div(f"❌ File gốc không tồn tại: {file_path}", style={"color": "red"})

    ds_time, ds_volt, boundaries = get_downsampled_data(file_path, max_points=5000)
    
    total_dur = boundaries[-1]["end_ms"] if boundaries else 0
    num_traces = len(boundaries)
    dt_ms = boundaries[0]["end_ms"] / len(ds_time) if len(ds_time) > 0 else 0.02
    fs_hz = 1000/dt_ms if dt_ms > 0 else 0

    label_defs = get_all_label_defs()
    dropdown_options = [{"label": l["name"], "value": l["code"]} for l in label_defs]

    return html.Div([
        dcc.Store(id="current-rec-id", data=rec_id),
        dcc.Store(id="current-file-path", data=str(file_path)),
        dcc.Store(id="current-boundaries", data=boundaries), 
        dcc.Store(id="current-dt-ms", data=dt_ms),
        
        # 1. INFO PANEL
        html.Div([
            html.Div([
                html.H4(f"{rec['full_name']} ({rec['patient_code']})", style={"margin": "0 0 5px 0", "color": "#1f2937"}),
                html.P(f"Ngày: {rec['visit_date']} | Giới tính: {rec.get('gender', 'N/A')}", style={"margin": 0, "color": "#6b7280"}),
                html.P(f"Bài đo: {rec['test_name']}", style={"margin": "5px 0 0 0", "fontWeight": "bold"})
            ], style={"flex": 1}),
            
            html.Div([
                html.P(f"Tần số mẫu: ~{int(fs_hz)} Hz", style={"margin": 0}),
                html.P(f"Tổng thời gian: {total_dur/1000:.1f} s", style={"margin": 0}),
                html.P(f"Số đoạn (Traces): {num_traces}", style={"margin": 0}),
            ], style={"flex": 0.5, "borderLeft": "1px solid #ccc", "paddingLeft": "15px", "fontSize": "13px", "color": "#4b5563"}),

            # FILTER CONTROLS
            html.Div([
                html.Label("Bộ lọc tín hiệu (DSP):", style={"fontWeight": "bold", "display": "block", "marginBottom": "5px"}),
                dcc.Checklist(
                    id="dsp-filters-checklist",
                    options=[
                        {'label': ' Khử nhiễu nguồn (Notch 50Hz)', 'value': 'NOTCH'},
                        {'label': ' Lọc thông dải (Bandpass 20-500Hz)', 'value': 'BANDPASS'},
                    ],
                    value=[],
                    labelStyle={'display': 'block', 'marginBottom': '3px'}
                )
            ], style={"backgroundColor": "#f0fdf4", "padding": "10px", "borderRadius": "8px", "border": "1px solid #86efac"}),
            
            html.Div(id="dsp-stats-display", style={"flex": 1, "textAlign": "right", "color": "#b91c1c", "fontWeight": "bold", "fontSize": "18px"})
        ], style={"backgroundColor": "white", "padding": "15px", "borderRadius": "8px", "display": "flex", "gap": "20px", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "marginBottom": "15px"}),

        # 2. GRAPH AREA
        dcc.Loading(
            dcc.Graph(
                id="analysis-graph", 
                style={"height": "55vh"},
                config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToAdd": ["drawrect", "eraseshape"]},
                figure=create_initial_figure(ds_time, ds_volt, boundaries)
            )
        ),
        
        # 3. ACTION BAR
        html.Div([
            html.Div([
                html.Label("Chọn loại nhãn:", style={"marginRight": "10px", "fontWeight": "bold"}),
                
                dcc.Dropdown(
                    id="label-type-dd",
                    options=dropdown_options,
                    value="PATH",
                    clearable=False,
                    searchable=True,
                    style={"width": "250px"}
                ),
                
                html.Button("➕", id="btn-open-modal", title="Tạo loại nhãn mới", 
                            style={"marginLeft": "5px", "padding": "5px 10px", "background": "#4b5563", "color": "white", "border": "none", "borderRadius": "4px"}),

                html.Button("Lưu vùng chọn", id="btn-save-label", 
                            style={"marginLeft": "15px", "padding": "8px 20px", "background": "#2563eb", "color": "white", "border": "none", "borderRadius": "4px", "fontWeight": "bold"}),
                
                html.Span(id="save-msg", style={"marginLeft": "15px", "color": "green", "fontWeight": "bold"})
            ], style={"display": "flex", "alignItems": "center"}),
            
             html.Button("📄 Xuất Báo Cáo PDF", id="btn-export-pdf", style={"padding": "8px 15px", "background": "#ea580c", "color": "white", "border": "none", "borderRadius": "4px"}),
             dcc.Download(id="download-pdf")

        ], style={"display": "flex", "justifyContent": "space-between", "padding": "10px", "backgroundColor": "#f9fafb", "marginTop": "10px", "borderRadius": "5px"}),

        html.Div(id="modal-create-label", children=[
            html.Div([
                html.H3("Tạo loại nhãn mới", style={"marginTop": 0}),
                html.Label("Mã nhãn (VD: MYO):"),
                dcc.Input(id="new-label-code", type="text", style={"width": "100%", "marginBottom": "10px", "padding": "5px"}),
                html.Label("Tên hiển thị (VD: Bệnh cơ):"),
                dcc.Input(id="new-label-name", type="text", style={"width": "100%", "marginBottom": "10px", "padding": "5px"}),
                html.Div([
                    html.Button("Hủy", id="btn-cancel-modal", style={"marginRight": "10px"}),
                    html.Button("Tạo mới", id="btn-confirm-modal", style={"background": "#2563eb", "color": "white", "border": "none", "padding": "5px 15px"})
                ], style={"textAlign": "right"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "8px", "width": "300px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"})
        ], style={"display": "none", "position": "fixed", "top": 0, "left": 0, "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)", "justifyContent": "center", "alignItems": "center", "zIndex": 1000}),

        # 4. LABELS TABLE
        html.H4("Danh sách vùng đã gán nhãn:", style={"marginTop": "20px"}),
        html.Div(id="labels-table-container"),

        dcc.Store(id="temp-stats-store")
    ])

def create_initial_figure(time, voltage, bounds):
    initial_range = [0, 5000] if len(time) > 0 else [0, 100]
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=time, y=voltage, mode='lines', line=dict(color='#9ca3af', width=1), name="Overview", hoverinfo='skip'))
    
    shapes = [dict(type="line", x0=b["start_ms"], x1=b["start_ms"], y0=0, y1=1, yref="paper", line=dict(color="rgba(0,0,0,0.1)")) for b in bounds[:100]]
    
    fig.update_layout(
        margin=dict(l=50, r=20, t=20, b=40), template="plotly_white",
        xaxis=dict(title="Time (ms)", range=initial_range),
        yaxis=dict(title="Amplitude (µV)", fixedrange=False),
        shapes=shapes, dragmode="pan"
    )
    return fig

# --- CALLBACKS ---

# 1. GRAPH LOGIC (Dynamic Load & Filters)
@app.callback(
    Output("analysis-graph", "figure"),
    Input("analysis-graph", "relayoutData"),
    Input("dsp-filters-checklist", "value"),
    State("analysis-graph", "figure"),
    State("current-file-path", "data"),
    State("current-dt-ms", "data"),
    prevent_initial_call=True
)
def update_graph_content(relayout, active_filters, fig, file_path, dt_ms):
    if not file_path: return no_update
    
    try: fig["layout"]["xaxis"]["rangeslider"]["yaxis"].pop("_template", None)
    except: pass

    if fig and "layout" in fig and "xaxis" in fig["layout"]:
        x0, x1 = fig["layout"]["xaxis"].get("range", [0, 100])
    else:
        x0, x1 = 0, 100

    if relayout:
        if "xaxis.range[0]" in relayout:
            x0, x1 = relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]
        elif "xaxis.range" in relayout:
            x0, x1 = relayout["xaxis.range"][0], relayout["xaxis.range"][1]
        
        # Chặn thời gian âm
        if x0 < 0: 
            diff = x1 - x0
            x0 = 0
            x1 = diff

    start_ms, end_ms = float(x0), float(x1)
    duration = end_ms - start_ms
    
    new_fig = go.Figure(fig)
    THRESHOLD_MS = 10000.0
    
    if duration < THRESHOLD_MS:
        pad = 100.0
        hr_time, hr_vals = get_data_slice(Path(file_path), start_ms - pad, end_ms + pad)
        
        # Filter
        processed_vals = hr_vals.copy()
        slice_dt = (hr_time[1] - hr_time[0]) if len(hr_time) > 1 else dt_ms
        
        if active_filters:
            if "NOTCH" in active_filters: processed_vals = apply_notch_filter(processed_vals, slice_dt)
            if "BANDPASS" in active_filters: processed_vals = apply_bandpass_filter(processed_vals, slice_dt)
            
        mask = (hr_time >= start_ms) & (hr_time <= end_ms)
        plot_time = hr_time[mask]
        plot_vals = processed_vals[mask]
        
        if len(new_fig.data) > 0:
            new_fig.data[0].x = plot_time
            new_fig.data[0].y = plot_vals
            new_fig.data[0].line.color = "#059669" if active_filters else "#1f2937"
            new_fig.data[0].name = "Detail (High-Res)"
    else:
        if len(new_fig.data) > 0:
            new_fig.data[0].line.color = "#9ca3af"
            new_fig.data[0].name = "Overview"

    new_fig.update_layout(xaxis=dict(range=[start_ms, end_ms]))
    return new_fig

# 2. STATS DISPLAY
@app.callback(
    Output("dsp-stats-display", "children"),
    Output("temp-stats-store", "data"),
    Input("analysis-graph", "relayoutData"),
    State("current-file-path", "data"),
    State("dsp-filters-checklist", "value"),
    prevent_initial_call=True
)
def update_stats(relayout, file_path, active_filters):
    if not relayout or "shapes" not in relayout: return no_update, no_update
    
    shape = relayout["shapes"][-1]
    x0, x1 = shape["x0"], shape["x1"]
    s, e = sorted([float(x0), float(x1)])
    
    t, v = get_data_slice(Path(file_path), s, e)
    if active_filters:
        dt = (t[1]-t[0]) if len(t)>1 else 0.02
        if "NOTCH" in active_filters: v = apply_notch_filter(v, dt)
        if "BANDPASS" in active_filters: v = apply_bandpass_filter(v, dt)
        
    stats = calculate_clinical_stats(t, v)
    return f"P2P: {stats['p2p_uv']}µV | RMS: {stats['rms_uv']}µV", {"start": s, "end": e, "stats": stats}

# 4. EXPORT PDF & ROUTING
@app.callback(
    Output("download-pdf", "data"),
    Input("btn-export-pdf", "n_clicks"),
    State("current-rec-id", "data"),
    State("current-file-path", "data"),
    prevent_initial_call=True
)
def export_pdf(n, rec_id, path):
    rec = get_recording_by_id(rec_id)
    labels = get_labels_by_recording(rec_id)
    labels_data = []
    for l in labels:
        t, v = get_data_slice(Path(path), l['start_ms'], l['end_ms'])
        labels_data.append({
            "trace_id": l['trace_id'], "start_ms": l['start_ms'], "end_ms": l['end_ms'],
            "label_type": l['label_type'], "stats": {"p2p_uv": l['p2p_uv'], "rms_uv": l['rms_uv'], "duration_ms": l['end_ms']-l['start_ms']},
            "time_arr": t, "volt_arr": v
        })
    pdf = generate_pdf_buffer({"full_name": rec['full_name'], "patient_code": rec['patient_code']}, rec, labels_data)
    if pdf: return dcc.send_bytes(pdf.getvalue(), f"Report_{rec['patient_code']}.pdf")
    return no_update

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(path):
    if path == "/": return layout_home()
    if path.startswith("/analysis/"): return layout_analysis(path.split("/")[-1])
    return "404"

# 5. UPLOAD & SEARCH
@app.callback(
    Output('upload-status', 'children'),
    Output('table-recordings', 'data'),
    Input('upload-data', 'contents'),
    Input('search-input', 'value'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_table(contents, search, filenames):
    ctx = callback_context
    if ctx.triggered and "upload-data" in ctx.triggered[0]['prop_id'] and contents:
        save_dir = get_base_path() / "data_raw" / "imported"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for content, name in zip(contents, filenames):
            try:
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                text = decoded.decode('utf-16')
                
                data = parse_natus_content(text)
                p_info = data['patient_info']
                
                p_id = add_patient_if_not_exists(p_info.get('patient_id', 'UNK'), p_info.get('first_name', 'Unknown'))
                
                safe_name = f"{int(time.time())}_{name}"
                f_path = save_dir / safe_name
                with open(f_path, "w", encoding="utf-16") as f: f.write(text)
                
                add_recording(p_id, p_info.get('visit_date'), p_info.get('test_name'), str(f_path))
            except Exception as e: print(e)
            
    # Handle Search & Refresh
    all_recs = get_all_recordings()
    rows = [{"id": r["id"], "date": r["visit_date"], "patient": f"{r['full_name']} ({r['patient_code']})", "test": r["test_name"], "action": f"[Xem chi tiết](/analysis/{r['id']})"} for r in all_recs]
    
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r['patient'].lower()]
        
    return " File dữ liệu này đã được xử lý, hãy kiểm tra danh sách", rows

@app.callback(
    Output("modal-create-label", "style"),
    Output("label-type-dd", "options"),
    Output("label-type-dd", "value"),
    Input("btn-open-modal", "n_clicks"),
    Input("btn-cancel-modal", "n_clicks"),
    Input("btn-confirm-modal", "n_clicks"),
    State("new-label-code", "value"),
    State("new-label-name", "value"),
    State("label-type-dd", "options"),
    prevent_initial_call=True
)
def handle_modal(n_open, n_cancel, n_confirm, code, name, current_options):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id']
    
    show_style = {"display": "flex", "position": "fixed", "top": 0, "left": 0, "width": "100%", "height": "100%", "backgroundColor": "rgba(0,0,0,0.5)", "justifyContent": "center", "alignItems": "center", "zIndex": 1000}
    hide_style = {"display": "none"}

    if "btn-open-modal" in trigger_id:
        return show_style, no_update, no_update
    
    if "btn-confirm-modal" in trigger_id:
        if code and name:
            success, msg = add_label_def(code.upper(), name)
            if success:
                new_options = current_options + [{"label": name, "value": code.upper()}]
                return hide_style, new_options, code.upper()
    
    return hide_style, no_update, no_update

# Xử lý Lưu Nhãn & Xóa Nhãn
@app.callback(
    Output("save-msg", "children"),
    Output("labels-table-container", "children"),
    Output("analysis-graph", "figure", allow_duplicate=True),
    Input("btn-save-label", "n_clicks"),
    Input("current-rec-id", "data"),
    Input({'type': 'del-btn', 'index': ALL}, 'n_clicks'),
    State("analysis-graph", "figure"),
    State("label-type-dd", "value"),
    State("current-file-path", "data"),
    State("current-boundaries", "data"),
    State("dsp-filters-checklist", "value"),
    prevent_initial_call=True
)
def batch_save_and_manage(n_save, rec_id, n_del, fig, label_type, file_path, boundaries, active_filters):
    ctx = callback_context
    if not ctx.triggered: return no_update, no_update, no_update
    trigger_id = ctx.triggered[0]['prop_id']
    
    msg = ""
    updated_fig = no_update

    # CASE A: Bấm nút LƯU
    if "btn-save-label" in trigger_id and fig and "layout" in fig:
        shapes = fig["layout"].get("shapes", [])

        user_drawn_shapes = [
            s for s in shapes 
            if s.get('type') == 'rect' and s.get('yref') != 'paper'
        ]
        
        if not user_drawn_shapes:
            msg = "Hãy vẽ ít nhất một vùng chọn trước!"
        else:
            count = 0
            for shape in user_drawn_shapes:
                x0, x1 = shape.get("x0"), shape.get("x1")
                if x0 is None or x1 is None: continue
                
                s, e = sorted([float(x0), float(x1)])
                
                t_arr, v_arr = get_data_slice(Path(file_path), s, e)
                
                dt = (t_arr[1]-t_arr[0]) if len(t_arr)>1 else 0.02
                if active_filters:
                    if "NOTCH" in active_filters: v_arr = apply_notch_filter(v_arr, dt)
                    if "BANDPASS" in active_filters: v_arr = apply_bandpass_filter(v_arr, dt)
                
                stats = calculate_clinical_stats(t_arr, v_arr)
                
                mid = (s+e)/2
                tid = "Unknown"
                for b in boundaries:
                    if b["start_ms"] <= mid < b["end_ms"]: tid = b["trace_id"]; break
                
                save_label_to_db(rec_id, s, e, tid, label_type, stats.get("p2p_uv",0), stats.get("rms_uv",0))
                count += 1
            
            if count > 0:
                msg = f"Đã lưu thành công {count} vùng chọn!"
            
                import copy
                updated_fig = copy.deepcopy(fig)
                updated_fig["layout"]["shapes"] = [
                    s for s in shapes
                    if s.get('type') != 'rect' or s.get('yref') == 'paper'
                ]
            else:
                msg = "Không tìm thấy vùng chọn hợp lệ"

    if "del-btn" in trigger_id:
        import json
        btn_dict = json.loads(trigger_id.split('.')[0])
        delete_label_by_id(btn_dict['index'])
        msg = "Đã xóa nhãn."

    labels = get_labels_by_recording(rec_id)
    if not labels:
        table = html.P("Chưa có nhãn nào.", style={"color": "gray"})
    else:
        all_defs = {d['code']: d for d in get_all_label_defs()}
        
        header = html.Tr([
            html.Th("Trace"), 
            html.Th("Start"), 
            html.Th("End"), 
            html.Th("Loại"), 
            html.Th("P2P"), 
            html.Th("RMS"), 
            html.Th("Xóa")
        ])
        rows = []
        for l in labels:
            lbl_def = all_defs.get(l['label_type'], {"name": l['label_type'], "color": "black"})
            
            rows.append(html.Tr([
                html.Td(l['trace_id']),
                html.Td(f"{l['start_ms']:.1f}"),
                html.Td(f"{l['end_ms']:.1f}"),
                html.Td(lbl_def['name'], style={"color": lbl_def['color'], "fontWeight": "bold"}),
                html.Td(f"{l['p2p_uv']}"),
                html.Td(f"{l['rms_uv']}"),
                html.Td(html.Button("❌", id={'type': 'del-btn', 'index': l['id']}, style={"border":"none", "background":"transparent", "cursor":"pointer"}))
            ]))
        table = html.Table([header] + rows, style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #ddd"})

    return msg, table, updated_fig

if __name__ == "__main__":
    app.run(debug=False, port=8051)
