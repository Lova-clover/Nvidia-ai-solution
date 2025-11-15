import numpy as np

from config import (
    MAX_AGE,
    IOU_MATCH,
    CENTER_GATE_SCALE,
)

class KalmanFilter2D:
    def __init__(self, x, y, dt: float = 1.0):
        self.x = np.array([[x],[y],[0.],[0.]], dtype=float)
        self.F = np.array([[1.,0.,dt,0.],
                           [0.,1.,0.,dt],
                           [0.,0.,1.,0. ],
                           [0.,0.,0.,1. ]], dtype=float)
        self.H = np.array([[1.,0.,0.,0.],
                           [0.,1.,0.,0.]], dtype=float)
        self.P = np.eye(4, dtype=float)*100.0
        self.R = np.eye(2, dtype=float)*5.0
        self.Q = np.eye(4, dtype=float)*0.01
        self.I = np.eye(4, dtype=float)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z: (2,1) [cx, cy]
        y = z - (self.H @ self.x)                     # (2,1)
        S = self.H @ self.P @ self.H.T + self.R       # (2,2)
        K = self.P @ self.H.T @ np.linalg.inv(S)      # (4,2)
        self.x = self.x + K @ y                       # (4,1)
        self.P = (self.I - K @ self.H) @ self.P       # (4,4)


class Track:
    def __init__(self, track_id, bbox, global_cls, conf):
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        self.kf = KalmanFilter2D(cx, cy)
        self.w  = float(x2-x1)
        self.h  = float(y2-y1)
        self.track_id = track_id
        self.time_since_update = 0
        self.hits = 1
        self.last_bbox = bbox
        self.last_global_cls = global_cls
        self.conf_ema = float(conf)

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox, global_cls, conf):
        x1,y1,x2,y2 = bbox
        cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
        self.kf.update(np.array([[cx],[cy]], dtype=float))
        w_meas, h_meas = float(x2-x1), float(y2-y1)
        self.w = 0.7*self.w + 0.3*w_meas
        self.h = 0.7*self.h + 0.3*h_meas
        self.time_since_update = 0
        self.hits += 1

        if global_cls == self.last_global_cls:
            self.conf_ema = 0.7*self.conf_ema + 0.3*float(conf)
        else:
            self.conf_ema = 0.85*self.conf_ema
        self.last_bbox = bbox
        self.last_global_cls = global_cls

    def get_bbox(self):
        cx, cy = float(self.kf.x[0,0]), float(self.kf.x[1,0])
        x1 = int(cx - self.w/2.0)
        y1 = int(cy - self.h/2.0)
        x2 = int(cx + self.w/2.0)
        y2 = int(cy + self.h/2.0)
        return x1,y1,x2,y2

    def get_state(self):
        x1,y1,x2,y2 = self.get_bbox()
        return x1,y1,x2,y2, self.last_global_cls, float(self.conf_ema), self.track_id, self.hits, self.last_bbox


def iou_box(b1,b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    w=max(0,x2-x1); h=max(0,y2-y1)
    inter=w*h
    if inter<=0: return 0.0
    a1=max(1,(b1[2]-b1[0])*(b1[3]-b1[1]))
    a2=max(1,(b2[2]-b2[0])*(b2[3]-b2[1]))
    return inter/float(a1+a2-inter)

def center_of(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def assign_detections_to_tracks(tracks, detections, iou_threshold=IOU_MATCH):
    if not tracks: return [],[],list(range(len(detections)))
    if not detections: return [],list(range(len(tracks))),[]
    M=np.zeros((len(tracks),len(detections)))
    for ti,t in enumerate(tracks):
        tb = t.get_bbox()
        tcx,tcy = center_of(tb)
        tw = max(1, tb[2]-tb[0]); th = max(1, tb[3]-tb[1])
        gate = CENTER_GATE_SCALE * max(tw, th)
        for di,d in enumerate(detections):
            db = d[:4]
            dcx,dcy = center_of(db)
            if np.hypot(tcx-dcx, tcy-dcy) > gate:
                M[ti,di] = 0.0
            else:
                M[ti,di] = iou_box(tb, db)
    matched=[]
    while True:
        idx = int(np.argmax(M))
        if M.flat[idx] < iou_threshold: break
        ti,di = divmod(idx, M.shape[1])
        matched.append((ti,di))
        M[ti,:] = -1; M[:,di] = -1
    mt=[m[0] for m in matched]; md=[m[1] for m in matched]
    ut=[i for i in range(len(tracks)) if i not in mt]
    ud=[i for i in range(len(detections)) if i not in md]
    return matched,ut,ud

class MultiObjectTracker:
    def __init__(self):
        self.tracks=[]; self.next_id=0
    def update(self, detections):
        for t in self.tracks: t.predict()
        matches,ut,ud = assign_detections_to_tracks(self.tracks, detections)
        for ti,di in matches:
            x1,y1,x2,y2,global_cls,conf = detections[di]
            self.tracks[ti].update((x1,y1,x2,y2), global_cls, conf)
        for di in ud:
            x1,y1,x2,y2,global_cls,conf = detections[di]
            self.tracks.append(Track(self.next_id, (x1,y1,x2,y2), global_cls, conf))
            self.next_id+=1
        self.tracks=[t for t in self.tracks if t.time_since_update<=MAX_AGE]
        out=[]
        for t in self.tracks:
            if t.time_since_update==0:
                out.append(t.get_state())
        return out
