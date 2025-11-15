import cv2
import numpy as np

from config import (
    USE_CALORIE,
    KCAL_PER100,
    BASE_GRAMS,
    GEOM,
)

def find_aruco_scale_cm_per_px(frame_bgr, marker_size_cm=5.0):
    if not USE_CALORIE:
        return None
    try:
        aruco = cv2.aruco
    except AttributeError:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dict4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:
        params = aruco.DetectorParameters_create()
    except AttributeError:
        try:
            params = aruco.DetectorParameters()
        except Exception:
            return None
    corners, ids, _ = aruco.detectMarkers(gray, dict4, parameters=params)
    if ids is None or len(corners)==0:
        return None
    max_side=0.0
    for c in corners:
        pts=c[0]
        edges=[np.linalg.norm(pts[(i+1)%4]-pts[i]) for i in range(4)]
        side=float(np.mean(edges)); max_side=max(max_side, side)
    if max_side<=0:
        return None
    return marker_size_cm / max_side

def estimate_calories(global_cls, bbox, cm_per_px):
    kcal100 = KCAL_PER100[global_cls]
    if cm_per_px is None:
        grams = BASE_GRAMS[global_cls]
    else:
        x1,y1,x2,y2 = bbox
        wpx=max(1,x2-x1); hpx=max(1,y2-y1)
        wcm = wpx*cm_per_px; hcm = hpx*cm_per_px
        thick = GEOM[global_cls]["thick_cm"]
        dens  = GEOM[global_cls]["density"]
        ksh   = GEOM[global_cls]["shape_k"]
        vol = wcm*hcm*thick*ksh
        grams = max(1.0, vol*dens)
    kcal = kcal100*(grams/100.0)
    return grams, kcal
