import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_fit_engine import run_ml_challenge

# ==========================================
# 📅 TODAY'S PUZZLE CONFIG
# ==========================================
DATE = "2026-04-17"
NAME = "The Waveform Jump"

def today_target(x):
    """
    Define today's mathematical puzzle here.
    Example: A combination of Sine and absolute values.
    """
    return torch.sin(2 * torch.pi * x) + 0.5 * torch.abs(x)
# ==========================================

if __name__ == "__main__":
    # Launch the game engine with today's settings
    run_ml_challenge(DATE, NAME, today_target)
