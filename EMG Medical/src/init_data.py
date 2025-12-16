import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db_manager import init_db, add_patient_if_not_exists, add_recording
from src.processing.loader import load_natus_txt

# 1. Khởi tạo bảng
init_db()

# 2. Tìm file mẫu
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data_raw" / "Le Thi Bop.txt"

if RAW_FILE.exists():
    print(f"🔄 Processing file: {RAW_FILE.name}...")
    
    # Đọc metadata bằng loader cũ
    data = load_natus_txt(RAW_FILE)
    p_info = data['patient_info']
    
    # Lưu vào DB
    # A. Tạo bệnh nhân
    p_id = add_patient_if_not_exists(
        patient_code=p_info.get('patient_id', 'UNKNOWN'),
        full_name=p_info.get('first_name', 'Unknown Patient')
    )
    
    # B. Tạo bản ghi
    rec_id = add_recording(
        patient_id=p_id,
        visit_date=p_info.get('visit_date'),
        test_name=p_info.get('test_name'),
        file_path=str(RAW_FILE), # Lưu đường dẫn tuyệt đối hoặc tương đối
        duration_ms=0 # Tạm thời để 0 hoặc tính từ data
    )
    
    print(f"✅ Successfully imported data for patient: {p_info.get('first_name')}")
else:
    print(f"⚠️ File not found: {RAW_FILE}")