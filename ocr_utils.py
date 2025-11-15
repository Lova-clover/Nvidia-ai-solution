import os
import tempfile
import re

import cv2

from models import ocr

def _run_paddle_ocr(image_bgr):
    """
    PaddleOCR 3.x 파이프라인 기준:
    - 항상 일정 이상 크게 스케일업해서 OCR
    - predict() 결과에서 rec_texts만 뽑아 텍스트 라인 리스트로 반환
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    # 짧은 변이 최소 256px 되도록 스케일업 (텍스트 인식률 강화)
    h, w = image_bgr.shape[:2]
    TARGET_SHORT = 256
    min_side = min(h, w)
    if min_side < TARGET_SHORT:
        scale = TARGET_SHORT / max(1, min_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 1차 시도: numpy 배열 그대로 predict
    try:
        result = ocr.predict(image_bgr)
    except Exception:
        # 2차 시도: 임시 파일로 저장 후 경로 기반 predict
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(tmp_path, image_bgr)
            result = ocr.predict(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    lines = []
    if not result:
        return lines

    for page in result:
        data = page.get("res", page) if isinstance(page, dict) else getattr(page, "res", None)
        if not isinstance(data, dict):
            continue
        texts = data.get("rec_texts", [])
        for t in texts:
            if isinstance(t, str) and t.strip():
                lines.append(t.strip())
    return lines


def ocr_text_from_crop(frame_bgr, bbox):
    """
    확정된 Drink(병/컵) 박스를 여러 패딩으로 크롭해서
    - 각 크롭을 좌우반전(거울 프레임 복원)
    - _run_paddle_ocr()로 OCR 수행
    - 텍스트 라인을 가장 많이 뽑은 버전을 최종 결과로 선택
    """
    if ocr is None:
        return []

    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox

    best_lines = []
    best_len   = 0

    # 패딩 3단계: 좁게/보통/넓게
    for pad in (20, 60, 120):
        xx1 = max(0, x1 - pad)
        yy1 = max(0, y1 - pad)
        xx2 = min(W, x2 + pad)
        yy2 = min(H, y2 + pad)

        if xx2 <= xx1 or yy2 <= yy1:
            continue

        crop = frame_bgr[yy1:yy2, xx1:xx2]
        if crop.size == 0:
            continue

        # 프레임이 거울 모드라 crop만 다시 좌우반전해서 텍스트 방향 복원
        crop = cv2.flip(crop, 1)

        # OCR 수행
        lines = _run_paddle_ocr(crop)
        lines = [l for l in lines if l.strip()]

        if len(lines) > best_len:
            best_len   = len(lines)
            best_lines = lines

        # 어느 정도 읽혔으면(3줄 이상) 더 넓은 패딩은 굳이 안 봐도 됨
        if best_len >= 3:
            break

    return best_lines

def parse_nutrient(text, key):
    """'당류 12 g' 형태에서 숫자 파싱."""
    m = re.search(rf"{key}\s*([\d\.]+)\s*g", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(rf"{key}[^\d]*([\d\.]+)\s*g", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return None

def categorize_drink(text):
    """텍스트에서 음료 타입/제로 여부 대략 분류."""
    t = text.lower()

    zero_keywords = ["제로", "무가당", "무설탕", "sugar free", "0kcal", "0 kcal", "zero sugar"]
    is_zero = any(k in t for k in zero_keywords)

    coffee_sweet_kw = ["라떼", "latte", "모카", "mocha", "카라멜", "caramel", "프라푸치노", "frappuccino", "바닐라", "돌체", "dolce"]
    coffee_plain_kw = ["아메리카노", "americano", "에스프레소", "espresso", "롱블랙", "long black"]
    coffee_general_kw = ["커피", "coffee", "카페", "cafe"]

    soda_kw = ["콜라", "coke", "coca-cola", "사이다", "sprite", "soda", "탄산음료"]
    juice_kw = ["주스", "juice", "에이드", "ade", "오렌지", "orange juice", "apple juice"]
    tea_kw   = ["녹차", "green tea", "홍차", "black tea", "우롱차", "oolong", "tea", "티백"]
    energy_kw = ["에너지 드링크", "energy drink", "몬스터", "monster", "레드불", "red bull", "burn"]
    water_kw  = ["생수", "bottled water", "water", "워터", "미네랄 워터"]

    for k in energy_kw:
        if k in t:
            return "energy", is_zero, k
    for k in coffee_sweet_kw:
        if k in t:
            return "coffee_sweet", is_zero, k
    for k in coffee_plain_kw:
        if k in t:
            return "coffee_plain", is_zero, k
    for k in coffee_general_kw:
        if k in t:
            return "coffee", is_zero, k
    for k in soda_kw:
        if k in t:
            return "soda", is_zero, k
    for k in juice_kw:
        if k in t:
            return "juice", is_zero, k
    for k in tea_kw:
        if k in t:
            return "tea", is_zero, k
    for k in water_kw:
        if k in t:
            return "water", is_zero, k

    return "unknown", is_zero, ""

def analyze_drink_nutrition(lines):
    """음료 OCR 텍스트 기반 당뇨 위험 분석."""
    if not lines:
        return (
            "OCR로 텍스트를 거의 읽지 못했습니다. "
            "라벨/성분표가 정면으로, 가까이 보이도록 다시 찍어주세요."
        )

    text = " ".join(lines)
    lower = text.lower()

    drink_type, is_zero, label_word = categorize_drink(text)

    sugar = parse_nutrient(text, "당류")
    carb  = parse_nutrient(text, "탄수화물")
    trans = parse_nutrient(text, "트랜스지방")

    msgs = []

    # 1) 타입 기반 멘트
    msgs.append("[음료 종류 추정]")
    if drink_type == "coffee_sweet":
        msgs.append(f" - 라떼/모카/카라멜 계열 커피로 보입니다. (키워드: {label_word})")
    elif drink_type == "coffee_plain":
        msgs.append(f" - 아메리카노/블랙 커피 계열로 보입니다. (키워드: {label_word})")
    elif drink_type == "coffee":
        msgs.append(f" - 커피 음료로 보이지만 단맛 정도는 라벨만으로 애매합니다. (키워드: {label_word})")
    elif drink_type == "soda":
        msgs.append(f" - 콜라/사이다 등 탄산음료로 보입니다. (키워드: {label_word})")
    elif drink_type == "juice":
        msgs.append(f" - 주스/에이드 계열로 보입니다. (키워드: {label_word})")
    elif drink_type == "tea":
        msgs.append(f" - 차(tea) 계열 음료로 보입니다. (키워드: {label_word})")
    elif drink_type == "energy":
        msgs.append(f" - 에너지 드링크 계열로 보입니다. (키워드: {label_word})")
    elif drink_type == "water":
        msgs.append(f" - 물/워터 계열로 보입니다. (키워드: {label_word})")
    else:
        msgs.append(" - 특정 음료 타입을 라벨에서 명확히 찾기 어렵습니다.")

    if is_zero:
        msgs.append(" - '제로/무가당/무설탕/sugar free/0kcal' 표기가 있습니다. → 당은 적을 가능성이 높습니다.")
    msgs.append("")

    # 2) 숫자 기반 분석
    msgs.append("[영양/당뇨 위험 평가]")

    used_numeric = False

    if sugar is not None:
        used_numeric = True
        if sugar >= 20:
            msgs.append(f" - 당류 {sugar:.1f} g → 한 번 마실 때 당이 상당히 많습니다. 당뇨 환자는 웬만하면 피하는 게 좋습니다.")
        elif sugar >= 10:
            msgs.append(f" - 당류 {sugar:.1f} g → 꽤 높은 편입니다. 마시더라도 반 컵 이하로 제한하는 게 낫습니다.")
        elif sugar >= 5:
            msgs.append(f" - 당류 {sugar:.1f} g → 중간 정도. 하루 전체 당 섭취량을 합산해서 관리해야 합니다.")
        else:
            msgs.append(f" - 당류 {sugar:.1f} g → 비교적 낮은 편입니다.")

    if carb is not None:
        used_numeric = True
        if carb >= 30:
            msgs.append(f" - 탄수화물 {carb:.1f} g → 전체 탄수화물 부담이 큰 편입니다.")
        elif carb >= 15:
            msgs.append(f" - 탄수화물 {carb:.1f} g → 중간 정도 탄수화물 양입니다.")

    if trans is not None:
        used_numeric = True
        if trans > 0.5:
            msgs.append(f" - 트랜스지방 {trans:.2f} g → 좋지 않은 지방이 꽤 있습니다. 자주 마시는 건 비추입니다.")
        elif trans > 0:
            msgs.append(f" - 트랜스지방 {trans:.2f} g → 소량이지만, 가능하면 트랜스지방 0 제품이 더 좋습니다.")
        else:
            msgs.append(" - 트랜스지방 0 g 표기가 있다면 지방 측면에서는 비교적 안심해도 됩니다.")

    # 3) 숫자 못 뽑았을 때 타입 기반 멘트
    if not used_numeric:
        if drink_type in ["soda", "juice", "coffee_sweet", "energy"]:
            if is_zero:
                msgs.append(" - 제로/무가당 버전이라면 일반 버전보다는 훨씬 낫지만, 카페인/인공감미료는 따로 고려해야 합니다.")
            else:
                msgs.append(" - 설탕/시럽이 많이 들어갔을 가능성이 큰 음료입니다. 당뇨 환자는 되도록 피하거나 극소량만 마시는 걸 추천합니다.")
        elif drink_type in ["coffee_plain", "tea", "water"]:
            msgs.append(" - 일반적으로 당이 거의 없거나 매우 적은 편입니다. 다른 간식/식사와 합쳐서 전체 탄수화물만 관리하면 됩니다.")
        else:
            msgs.append(" - 정확한 영양값을 읽지 못해서 타입만 가지고 추정했습니다. 성분표가 잘 보이도록 한 번 더 찍어주는 게 좋습니다.")

    return "\n".join(msgs)
