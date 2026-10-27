import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.metrics import compute_metrics
import numpy as np

def main():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.6, 0.9])
    m = compute_metrics(y_true, y_pred, y_prob)
    print(m)

if __name__ == "__main__":
    main()