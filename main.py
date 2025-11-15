import cv2

from config import (
    FOOD_NAMES,
    RISK_MESSAGES,
    USE_CALORIE,
    FONT,
)
from detection import detect_step
from ocr_utils import ocr_text_from_crop, analyze_drink_nutrition
from calorie import find_aruco_scale_cm_per_px, estimate_calories


def run_camera():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "카메라 열기 실패"

    cm_per_px = None
    frame_idx = 0
    final_msg = None

    print("[INFO] 라이브 프리뷰 시작. 확정되면 창 닫고 콘솔에 결과 출력. (종료: q)")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] 프레임 캡처 실패. 종료.")
            break

        # 거울 모드
        frame = cv2.flip(frame, 1)
        frame_idx += 1

        frame_vis, conf_state = detect_step(frame, frame_idx)
        cv2.imshow("Food/Drink Detection", frame_vis)

        if conf_state is not None:
            global_cls, bbox, cema = conf_state
            lines = []
            lines.append(f"[CONFIRMED] {FOOD_NAMES[global_cls]}  ema={cema:.2f}")
            lines.append(f"[RISK] {RISK_MESSAGES.get(global_cls, '해당 음식/음료는 당뇨 관리에 주의가 필요합니다.')}")

            # OCR 시도
            ocr_lines = ocr_text_from_crop(frame, bbox)
            if ocr_lines:
                lines.append("[OCR TEXT (상위 일부)]")
                for t in ocr_lines[:6]:
                    lines.append(f" - {t}")
            else:
                lines.append("[OCR] 포장/텍스트를 거의 인식하지 못했습니다.")

            # 음료라면 영양 성분 분석
            if global_cls == 2:
                nutr_msg = analyze_drink_nutrition(ocr_lines)
                if nutr_msg:
                    lines.append("[OCR 기반 음료 영양 분석]")
                    for row in nutr_msg.split("\n"):
                        lines.append(f" - {row}")

            # 칼로리 러프 추정
            if USE_CALORIE:
                if cm_per_px is None:
                    cm_per_px = find_aruco_scale_cm_per_px(frame_vis)
                    if cm_per_px is not None:
                        lines.append(f"[INFO] 스케일 감지: {cm_per_px:.4f} cm/px")
                grams, kcal = estimate_calories(global_cls, bbox, cm_per_px)
                lines.append("[CALORIE EST]")
                lines.append(f" - grams : {grams:.1f} g")
                lines.append(f" - kcal  : {kcal:.1f} kcal")

            final_msg = "\n".join(lines)
            break

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if final_msg:
        print(final_msg)
    else:
        print("[INFO] 최종 확정 결과 없음.")


if __name__ == "__main__":
    run_camera()
