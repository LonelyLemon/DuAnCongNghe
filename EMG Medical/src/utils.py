import sys
import os
from pathlib import Path

def get_base_path():
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent.parent

def get_temp_path():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent