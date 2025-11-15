from ultralytics import YOLO
from paddleocr import PaddleOCR

from config import (
    FRUIT_MODEL_PATH,
    COCO_MODEL_PATH,
)

print("[INFO] YOLO fruit model:", FRUIT_MODEL_PATH)
print("[INFO] YOLO COCO model:", COCO_MODEL_PATH)

fruit_model = YOLO(FRUIT_MODEL_PATH)
COCO_MODEL  = YOLO(COCO_MODEL_PATH)

print("[INFO] Init PaddleOCR (korean)")
ocr = PaddleOCR(
    lang="korean",
    use_textline_orientation=True,       # 글자 방향 보정
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)
