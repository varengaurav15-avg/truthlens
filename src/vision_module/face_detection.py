from typing import List, Tuple
import numpy as np

def detect_faces(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        out = detector.detect_faces(img)
        return [tuple(box) for box in [o["box"] for o in out]]
    except Exception:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        rects = cascade.detectMultiScale(gray, 1.3, 5)
        return [tuple(map(int, (x, y, w, h))) for (x, y, w, h) in rects]