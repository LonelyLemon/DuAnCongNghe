import numpy as np
from scipy import signal

def apply_notch_filter(data, dt_ms, freq=50.0, quality_factor=30.0):
    """
    Lọc bỏ nhiễu nguồn điện (50Hz hoặc 60Hz).
    Sử dụng bộ lọc IIR Notch.
    """
    if len(data) == 0: return data
    
    # Tính tần số lấy mẫu (Sampling Frequency - Hz)
    fs = 1000.0 / dt_ms
    
    # Tạo bộ lọc
    # w0: Tần số cần loại bỏ (Normalized frequency)
    # Q: Quality factor - Q càng cao thì dải cắt càng hẹp (ít ảnh hưởng tín hiệu gốc)
    b, a = signal.iirnotch(w0=freq, Q=quality_factor, fs=fs)
    
    # Áp dụng bộ lọc (filtfilt giúp lọc 2 chiều để không bị lệch pha)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def apply_bandpass_filter(data, dt_ms, lowcut=20.0, highcut=500.0, order=4):
    """
    Lọc thông dải (Bandpass): Chỉ giữ lại tín hiệu trong khoảng lowcut -> highcut.
    Thường dùng cho EMG: 20Hz - 500Hz.
    Sử dụng bộ lọc Butterworth.
    """
    if len(data) == 0: return data
    
    fs = 1000.0 / dt_ms
    
    # Kiểm tra điều kiện Nyquist (Tần số cắt phải nhỏ hơn một nửa tần số lấy mẫu)
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        highcut = nyquist - 1.0 # Điều chỉnh an toàn
        
    # Tạo bộ lọc Butterworth
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    
    # Áp dụng
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data