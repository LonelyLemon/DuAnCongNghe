import os
import random
import re
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data_raw" / "Le Thi Bop.txt"
OUTPUT_DIR = BASE_DIR / "data_raw" / "mock_generated"
NUM_FILES_TO_GEN = 10

NAMES = ["Nguyen Van A", "Tran Thi B", "Le Van C", "Pham Thi D", "Hoang Van E", 
         "Do Thi F", "Vu Van G", "Dang Thi H", "Bui Van I", "Ngo Thi K"]

def generate_mock_data():
    if not INPUT_FILE.exists():
        print(f"Không tìm thấy file mẫu: {INPUT_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(INPUT_FILE, "r", encoding="utf-16") as f:
        content = f.read()

    print(f"Đang tạo {NUM_FILES_TO_GEN} file giả lập từ mẫu")

    for i in range(NUM_FILES_TO_GEN):
        new_content = content
        
        # 1. Fake Metadata
        new_name = NAMES[i % len(NAMES)]
        new_id = str(random.randint(1000000000, 9999999999))
        
        new_content = re.sub(r"Patient ID=.*", f"Patient ID={new_id}", new_content)
        new_content = re.sub(r"First Name=.*", f"First Name={new_name}", new_content)
        new_content = re.sub(r"Visit Date=.*", f"Visit Date={random.randint(1,28)}/{random.randint(1,12)}/2024 10:00:00 SA", new_content)

        # 2. Fake Signal Data 
        def noise_injector(match):
            raw_chunk = match.group(0)
            nums = re.findall(r"-?\d+,\d+|-?\d+", raw_chunk)
            if not nums: return raw_chunk
            
            new_chunk_parts = []
            for n_str in nums:
                try:
                    val = float(n_str.replace(",", "."))
                    noise = val * 0.1 * random.uniform(-1, 1)
                    new_val = val + noise
                    new_val_str = f"{new_val:.2f}".replace(".", ",")
                    new_chunk_parts.append(new_val_str)
                except:
                    new_chunk_parts.append(n_str)
            
            return ",".join(new_chunk_parts)
        
        out_name = f"Mock_{new_id}_{new_name.replace(' ', '')}.txt"
        out_path = OUTPUT_DIR / out_name
        
        with open(out_path, "w", encoding="utf-16") as f:
            f.write(new_content)
        
        print(f"✅ Đã tạo: {out_name}")

if __name__ == "__main__":
    generate_mock_data()