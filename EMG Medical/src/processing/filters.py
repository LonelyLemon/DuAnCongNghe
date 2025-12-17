import numpy as np
from scipy import signal

def apply_notch_filter(data, dt_ms, freq=50.0, quality_factor=30.0):
    if len(data) < 30:
        return data
    
    # Tính tần số lấy mẫu (Sampling Frequency - Hz)
    fs = 1000.0 / dt_ms
    
    b, a = signal.iirnotch(w0=freq, Q=quality_factor, fs=fs)
    
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def apply_bandpass_filter(data, dt_ms, lowcut=20.0, highcut=500.0, order=4):
    if len(data) < 30: return data
    
    fs = 1000.0 / dt_ms
    
    nyquist = 0.5 * fs
    if highcut >= nyquist:
        highcut = nyquist - 1.0
        
    b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)
    
    try:
        filtered_data = signal.filtfilt(b, a, data)
    except ValueError:
        return data
        
    return filtered_data