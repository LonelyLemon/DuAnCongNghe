import numpy as np

def calculate_clinical_stats(time_arr, voltage_arr):
    if len(voltage_arr) == 0:
        return {}

    # 1. Peak-to-Peak Amplitude
    v_max = np.max(voltage_arr)
    v_min = np.min(voltage_arr)
    p2p = v_max - v_min

    # 2. RMS (Root Mean Square)
    rms = np.sqrt(np.mean(voltage_arr**2))

    # 3. Duration
    duration = time_arr[-1] - time_arr[0] if len(time_arr) > 1 else 0

    # 4. Phases / Zero Crossings
    mean_val = np.mean(voltage_arr)
    centered = voltage_arr - mean_val
    zero_crossings = np.where(np.diff(np.signbit(centered)))[0].size

    return {
        "p2p_uv": round(p2p, 2),
        "rms_uv": round(rms, 2),
        "duration_ms": round(duration, 2),
        "turns": int(zero_crossings),
        "max_uv": round(v_max, 2),
        "min_uv": round(v_min, 2)
    }