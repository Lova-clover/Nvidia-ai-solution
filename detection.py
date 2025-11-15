import cv2
import numpy as np

from config import (
    FOOD_NAMES,
    FRUIT_CLASS_MAP,
    COCO_DRINK_CLASS_IDS,
    AR_RANGE,
    MIN_AREA_FRAC,
    CONF_THRES_FRUIT,
    CONF_THRES_DRINK,
    IOU_THRES,
    STARTUP_SKIP_FRAMES,
    MIN_HITS,
    EMA_ON_THRES,
    VERIFY_WINDOW,
    VERIFY_KEEP_MIN,
    STAB_IOU_MIN,
    STAB_CENTER_MAX_PX,
    REVERIFY_CROP,
    REVERIFY_PAD,
    REVERIFY_CONF,
    COOLDOWN_FRAMES,
    FONT,
)

from models import fruit_model, COCO_MODEL
from tracking import MultiObjectTracker, iou_box, center_of

tracker = MultiObjectTracker()
confirmed = None   # (global_cls, bbox, conf_ema)
candidate = None   # {'tid','cls','frames','agree','last_bbox','last_conf'}
cooldown_until = 0

def begin_candidate(tid, cls, bbox, conf_ema):
    return {
        'tid': tid,
        'cls': cls,
        'frames': 0,
        'agree': 0,
        'last_bbox': bbox,
        'last_conf': conf_ema,
    }

def reset_candidate():
    global candidate
    candidate = None

def detect_step(frame_bgr, frame_idx):
    """
    한 프레임에 대해
    - YOLO fruit + YOLO COCO(drink)
    - Kalman multi-object tracking
    - 디바운싱/재검증을 거쳐 확정 객체 결정
    """
    global confirmed, candidate, cooldown_until

    H, W = frame_bgr.shape[:2]
    detections = []

    # 1) 과일 YOLO 감지
    fruit_results = fruit_model.predict(
        source=frame_bgr,
        conf=CONF_THRES_FRUIT,
        iou=IOU_THRES,
        verbose=False
    )
    for r in fruit_results:
        for b in r.boxes:
            raw_cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            if raw_cls not in FRUIT_CLASS_MAP:
                continue
            gcls = FRUIT_CLASS_MAP[raw_cls]
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
            w,h = x2-x1, y2-y1
            if h <= 0 or w <= 0:
                continue
            area = w*h
            ar = w / float(max(1, h))
            if area < MIN_AREA_FRAC[gcls] * (W*H):
                continue
            lo,hi = AR_RANGE[gcls]
            if not (lo <= ar <= hi):
                continue
            conf = float(b.conf)
            detections.append((x1,y1,x2,y2, gcls, conf))

    # 2) COCO YOLO에서 음료 용기만 감지 → Drink(2)
    drink_results = COCO_MODEL.predict(
        source=frame_bgr,
        conf=CONF_THRES_DRINK,
        iou=IOU_THRES,
        verbose=False
    )
    for r in drink_results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            raw_cls = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            if raw_cls not in COCO_DRINK_CLASS_IDS:
                continue
            gcls = 2  # Drink
            x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
            w,h = x2-x1, y2-y1
            if h <= 0 or w <= 0:
                continue
            area = w*h
            ar = w / float(max(1, h))
            if area < MIN_AREA_FRAC[gcls] * (W*H):
                continue
            lo,hi = AR_RANGE[gcls]
            if not (lo <= ar <= hi):
                continue
            conf = float(b.conf)
            detections.append((x1,y1,x2,y2, gcls, conf))

    # 3) 트래커 업데이트
    tracked = tracker.update(detections)

    frame_draw = frame_bgr.copy()
    for x1,y1,x2,y2, gcls, conf_ema, tid, hits, last_bbox in tracked:
        label = f"ID{tid} {FOOD_NAMES[gcls]} ema={conf_ema:.2f} hits={hits}"
        cv2.rectangle(frame_draw, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame_draw, label,
            (x1, max(0,y1-8)),
            FONT, 0.6, (0,255,0), 2
        )

        first_pass = (hits >= MIN_HITS and conf_ema >= EMA_ON_THRES)

        if frame_idx < STARTUP_SKIP_FRAMES or frame_idx < cooldown_until:
            first_pass = False

        if first_pass:
            if candidate is None or candidate['tid'] != tid or candidate['cls'] != gcls:
                candidate = begin_candidate(tid, gcls, (x1,y1,x2,y2), conf_ema)
            else:
                lx1,ly1,lx2,ly2 = candidate['last_bbox']
                iou = iou_box((lx1,ly1,lx2,ly2), (x1,y1,x2,y2))
                (pcx,pcy) = center_of((lx1,ly1,lx2,ly2))
                (ccx,ccy) = center_of((x1,y1,x2,y2))
                center_shift = np.hypot(pcx-ccx, pcy-ccy)

                candidate['frames'] = min(candidate['frames']+1, VERIFY_WINDOW)
                if iou >= STAB_IOU_MIN or center_shift <= STAB_CENTER_MAX_PX:
                    candidate['agree'] = min(candidate['agree']+1, VERIFY_WINDOW)
                candidate['last_bbox'] = (x1,y1,x2,y2)
                candidate['last_conf'] = conf_ema

                cv2.putText(
                    frame_draw,
                    f"VERIFYING {candidate['agree']}/{candidate['frames']} (win {VERIFY_WINDOW})",
                    (12, 28),
                    FONT, 0.75, (0,0,255), 2
                )

                if candidate['frames'] >= VERIFY_WINDOW:
                    passed_majority = (candidate['agree'] >= VERIFY_KEEP_MIN)
                    ok_final = passed_majority

                    # 과일만 재검증. 음료(2)는 재검증 생략.
                    if REVERIFY_CROP and ok_final and gcls in (0, 1):
                        cx1,cy1,cx2,cy2 = candidate['last_bbox']
                        px1 = max(0, cx1-REVERIFY_PAD); py1 = max(0, cy1-REVERIFY_PAD)
                        px2 = min(W, cx2+REVERIFY_PAD); py2 = min(H, cy2+REVERIFY_PAD)
                        roi = frame_bgr[py1:py2, px1:px2]
                        if roi.size > 0:
                            rr = fruit_model.predict(
                                source=roi,
                                conf=REVERIFY_CONF,
                                iou=0.30,
                                verbose=False
                            )
                            vcls = None
                            for rrk in rr:
                                if len(rrk.boxes) == 0:
                                    continue
                                idx = int(np.argmax(rrk.boxes.conf.cpu().numpy()))
                                raw = int(rrk.boxes.cls[idx].item())
                                if raw in FRUIT_CLASS_MAP:
                                    vcls = FRUIT_CLASS_MAP[raw]
                                    break
                            if vcls is not None:
                                ok_final = ok_final and (vcls == gcls)

                    if ok_final:
                        confirmed = (gcls, candidate['last_bbox'], candidate['last_conf'])
                        reset_candidate()
                        break
                    else:
                        cooldown_until = frame_idx + COOLDOWN_FRAMES
                        reset_candidate()
        else:
            reset_candidate()

    if confirmed is not None:
        cid, _, cema = confirmed
        cv2.putText(
            frame_draw,
            f"CONFIRMED: {FOOD_NAMES[cid]} (ema={cema:.2f})",
            (12, 30),
            FONT,
            0.9,
            (0,0,255),
            3
        )

    return frame_draw, confirmed
