import os
import cv2

# ==========================
# 0) Paddle / PaddleX 캐시 경로
# ==========================
PADDLE_HOME = r"C:\paddle_ocr_home"
os.environ["PADDLE_HOME"] = PADDLE_HOME
os.environ["HOME"] = PADDLE_HOME
os.environ["USERPROFILE"] = PADDLE_HOME
os.environ["HOMEDRIVE"] = "C:"
os.environ["HOMEPATH"] = r"\paddle_ocr_home"
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "HF")
os.makedirs(PADDLE_HOME, exist_ok=True)

# ==========================
# 1) 모델 경로 및 공통 설정
# ==========================
FRUIT_MODEL_PATH  = r"C://Users//성주//OneDrive//바탕 화면//Nvidia ai 솔루션//best.pt" # 환경에 따라 변경 필요
COCO_MODEL_PATH   = "yolov8n.pt"

# 글로벌 클래스 id
# 0: Banana, 1: Watermelon, 2: Drink (병/컵/잔)
FOOD_NAMES = {
    0: "Banana",
    1: "Watermelon",
    2: "Drink",
}

RISK_MESSAGES = {
    0: "바나나는 당지수가 높아서 혈당을 빠르게 올릴 수 있습니다.",
    1: "수박은 수분이 많지만 당지수가 높아 대량 섭취 시 혈당을 급격히 올릴 수 있습니다.",
    2: "당 들어간 음료는 혈당을 급격히 올릴 수 있습니다. 라벨의 당류/탄수화물 양을 꼭 확인해야 합니다.",
}

# YOLO 결과 → 글로벌 클래스 매핑
# fruit_model: cls 0→Banana, 1→Watermelon
FRUIT_CLASS_MAP  = {0: 0, 1: 1}

# COCO 모델에서 음료 용기만 골라서 Drink(2)로 매핑
# 39: bottle, 40: wine glass, 41: cup
COCO_DRINK_CLASS_IDS = {39, 40, 41}

ALLOW_GLOBAL = [0, 1, 2]

# detection / tracking 파라미터
CONF_THRES_FRUIT = 0.85
CONF_THRES_DRINK = 0.45
IOU_THRES  = 0.50
MAX_AGE    = 10
IOU_MATCH  = 0.30
CENTER_GATE_SCALE = 0.6

# 클래스별 aspect ratio / 최소 면적 게이트
AR_RANGE = {
    0: (0.25, 6.00),  # Banana
    1: (0.60, 2.00),  # Watermelon
    2: (0.25, 4.00),  # Drink (병/컵 세로로 긴 것 허용)
}
MIN_AREA_FRAC = {
    0: 0.0015,
    1: 0.0020,
    2: 0.0015,
}

# 확정(디바운싱)
STARTUP_SKIP_FRAMES  = 10
MIN_HITS             = 8
EMA_ON_THRES         = 0.80
VERIFY_WINDOW        = 15
VERIFY_KEEP_MIN      = 11
STAB_IOU_MIN         = 0.50
STAB_CENTER_MAX_PX   = 40
REVERIFY_CROP        = True
REVERIFY_PAD         = 12
REVERIFY_CONF        = 0.88
COOLDOWN_FRAMES      = 20

# 칼로리 추정
USE_CALORIE = True
KCAL_PER100 = {
    0: 89.0,   # 바나나
    1: 30.0,   # 수박
    2: 45.0,   # 음료: 대충 당 들어간 음료 기준
}
BASE_GRAMS  = {
    0: 118.0,
    1: 150.0,
    2: 250.0,  # 음료 1병/컵 250ml 가정
}
GEOM = {
    0: {"thick_cm": 3.0, "density": 0.95, "shape_k": 0.60},
    1: {"thick_cm": 3.0, "density": 0.96, "shape_k": 0.80},
    2: {"thick_cm": 5.0, "density": 1.00, "shape_k": 0.50},
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
