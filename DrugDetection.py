import os
import sys

from models import ocr  

# 환경에 따라 변경 필요
DEFAULT_IMG_PATH = r"C://Users//성주//OneDrive//바탕 화면//Nvidia ai 솔루션//coffee.jpg"

def run_ocr_on_image(img_path: str):
    """주어진 이미지 경로에 대해 PaddleOCR 실행하고 텍스트 + 점수 출력."""
    if not os.path.exists(img_path):
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {img_path}")
        return

    print(f"[INFO] PaddleOCR 실행: {img_path}")
    try:
        result = ocr.predict(img_path)  # PaddleOCR 3.x 스타일
    except Exception as e:
        print("[ERROR] OCR 실행 중 예외 발생:")
        print(e)
        return

    any_text = False

    for i, page in enumerate(result):
        data = page.get("res", page) if isinstance(page, dict) else getattr(page, "res", None)
        if not isinstance(data, dict):
            continue

        texts  = data.get("rec_texts", [])
        scores = data.get("rec_scores", [None] * len(texts))

        print(f"\n=== PAGE {i} ===")
        for t, s in zip(texts, scores):
            if not t:
                continue
            any_text = True
            if s is None:
                print(t)
            else:
                print(f"{s:.3f}\t{t}")

    if not any_text:
        print("\n[WARN] 인식된 텍스트가 없습니다. 해상도/초점/조명을 다시 확인하세요.")


def main():
    # 사용법:
    #   python DrugDetection.py           → DEFAULT_IMG_PATH로 실행
    #   python DrugDetection.py path.jpg  → 지정한 이미지로 실행
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        img_path = DEFAULT_IMG_PATH
        print(f"[INFO] 이미지 경로 인자를 못 받았으므로 DEFAULT 경로로 실행합니다.\n"
              f"       수정하려면 DrugDetection.py의 DEFAULT_IMG_PATH를 바꾸든지,\n"
              f"       `python DrugDetection.py 실제경로` 로 실행하세요.")

    run_ocr_on_image(img_path)


if __name__ == "__main__":
    main()
