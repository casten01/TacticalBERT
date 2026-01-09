import numpy as np
import pandas as pd
from .config import Config

def discretize_location(x, y):
    """
    Converts continuous coordinates (0-120, 0-80) in a unique Token ID.
    Row-Major flattening.
    """
    if pd.isna(x) or pd.isna(y):
        return 0 # Token [PAD] / [UNK]
    
    x = np.clip(x, 0, Config.GRID_WIDTH - 1e-6)
    y = np.clip(y, 0, Config.GRID_HEIGHT - 1e-6)
    
    x_int = int(x)
    y_int = int(y)
    
    token_id = (y_int * Config.GRID_WIDTH) + x_int + 1
    return token_id

def discretize_duration(seconds):
    """
    Logaritmic Binning for the duration of an event.
    """
    if pd.isna(seconds):
        return 0 # [NONE]
    
    if seconds < 0.5: return 1
    if seconds < 1.0: return 2  
    if seconds < 1.5: return 3  
    if seconds < 2.0: return 4  
    if seconds < 3.0: return 5  
    if seconds < 4.0: return 6 
    if seconds < 6.0: return 7  
    if seconds < 10.0: return 8 
    return 9                    