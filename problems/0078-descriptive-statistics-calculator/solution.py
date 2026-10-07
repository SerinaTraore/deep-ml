import numpy as np
from collections import Counter

def descriptive_statistics(data: list | np.ndarray) -> dict:
    arr = np.asarray(data)
    
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    
    # Mode avec Counter (version pure Python)
    counts = Counter(arr.tolist())
    mode_val = counts.most_common(1)[0][0]
    
    variance_val = float(np.var(arr))
    std_val = float(np.std(arr))
    
    # Quartiles avec quantile
    p25 = float(np.quantile(arr, 0.25))
    p50 = float(np.quantile(arr, 0.50))
    p75 = float(np.quantile(arr, 0.75))
    iqr_val = float(p75 - p25)
    
    return {
        "mean": mean_val,
        "median": median_val,
        "mode": mode_val,
        "variance": variance_val,
        "standard_deviation": std_val,
        "25th_percentile": p25,
        "50th_percentile": p50,
        "75th_percentile": p75,
        "interquartile_range": iqr_val
    }