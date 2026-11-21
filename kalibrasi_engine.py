import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
# Tiga titik landmark per jari, sama seperti dokumentasi arsitektur
JOINTS = {
    'thumb': (2, 3, 4), 'index': (5, 6, 8), 'middle': (9, 10, 12),
    'ring': (13, 14, 16), 'pinky': (17, 18, 20),
}

# Rekomendasi awal sistem (usulan, bukan keputusan klinis): (durasi menit, level, haptic)
RECOMMENDATION = {
    'Kaku / Spastisitas Berat': (10, 'Easy', True),
    'Genggaman Terbatas Sedang': (15, 'Medium', True),
    'Genggaman Sempurna / Baik': (20, 'Hard', False),
}

def calculate_angle(a, b, c):
    # 0 derajat = jari lurus, makin besar = makin menekuk
    a, b, c = (np.array([p.x, p.y, p.z]) for p in (a, b, c))
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return round(max(0.0, 180.0 - angle), 1)

def classify_grip(avg_angle):
    if avg_angle >= 70.0:
        return 'Genggaman Sempurna / Baik', 'Pasien mampu mengepal penuh. Siap untuk latihan ketahanan dan kecepatan.'
    if avg_angle >= 40.0:
        return 'Genggaman Terbatas Sedang', 'Fleksi jari parsial. Direkomendasikan latihan peningkatan ROM bertahap.'
    return 'Kaku / Spastisitas Berat', 'Keterbatasan gerak signifikan. Wajib durasi singkat dengan Haptic Assist aktif.'

class RepCounter:
    # Histeresis relatif terhadap kemampuan pasien, tidak memakai angka tetap
    def __init__(self, min_peak=15.0, close_ratio=0.6, open_ratio=0.3):
        self.peak, self.state, self.count = 0.0, 'open', 0
        self.t_start, self.durations = None, []
        self.min_peak, self.close_ratio, self.open_ratio = min_peak, close_ratio, open_ratio

    def update(self, mean_flex, t):
        self.peak = max(self.peak, mean_flex)
        if self.peak < self.min_peak:
            return
        if self.state == 'open' and mean_flex >= self.close_ratio * self.peak:
            self.state, self.t_start = 'closed', t
        elif self.state == 'closed' and mean_flex <= self.open_ratio * self.peak:
            self.state = 'open'
            self.count += 1
            self.durations.append(round(t - self.t_start, 2))

class CalibrationEngine:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.lock = threading.Lock()
        self.running, self.error, self.frame, self._gen = False, None, None, 0
        self._reset_state()

    def _reset_state(self):
        self.max_angles = {f: 0.0 for f in FINGERS}
        self.current = {f: 0.0 for f in FINGERS}
        self.buf = {f: deque(maxlen=5) for f in FINGERS}
        self.counter = RepCounter()
        self.valid_frames = 0
        self.hand_visible = False

    def start(self):
        if not self.running:
            self._gen += 1
            self.running, self.error = True, None
            threading.Thread(target=self._loop, args=(self._gen,), daemon=True).start()

    def stop(self):
        self.running = False

    def reset(self):
        with self.lock:
            self._reset_state()

    def get_frame(self):
        with self.lock:
            return self.frame

    def snapshot(self):
        with self.lock:
            vals = list(self.max_angles.values())
            avg = round(sum(vals) / len(vals), 1)
            status, note = classify_grip(avg)
            durs = self.counter.durations
            return {
                'running': self.running, 'error': self.error, 'hand_visible': self.hand_visible,
                'valid_frames': self.valid_frames,
                'current': dict(self.current), 'max_angles': dict(self.max_angles),
                'avg_max': avg, 'grip_status': status, 'clinical_note': note,
                'movement': {
                    'weakest_finger': min(self.max_angles, key=self.max_angles.get),
                    'finger_spread_deg': round(max(vals) - min(vals), 1),
                    'repetitions': self.counter.count,
                    'mean_cycle_duration_s': round(float(np.mean(durs)), 2) if durs else None,
                },
            }

    def _loop(self, gen):
        hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.7)
        draw = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error = 'Kamera tidak dapat dibuka. Periksa izin kamera atau indeks kamera.'
        try:
            while self.running and gen == self._gen and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                visible = bool(res.multi_hand_landmarks and res.multi_hand_world_landmarks
                               and res.multi_handedness[0].classification[0].score >= 0.8)
                if visible:
                    lm = res.multi_hand_world_landmarks[0].landmark
                    with self.lock:
                        for f, (a, b, c) in JOINTS.items():
                            self.buf[f].append(calculate_angle(lm[a], lm[b], lm[c]))
                            self.current[f] = round(float(np.median(self.buf[f])), 1)
                            self.max_angles[f] = max(self.max_angles[f], self.current[f])
                        self.valid_frames += 1
                        self.counter.update(float(np.mean(list(self.current.values()))), time.time())
                    draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
                ok, jpg = cv2.imencode('.jpg', frame)
                with self.lock:
                    self.hand_visible = visible
                    if ok:
                        self.frame = jpg.tobytes()
        finally:
            cap.release()
            hands.close()
            if gen == self._gen:
                self.running, self.frame = False, None