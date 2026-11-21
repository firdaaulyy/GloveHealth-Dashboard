import time
from kalibrasi_engine import CalibrationEngine

e = CalibrationEngine()
e.start()
for _ in range(20):
    time.sleep(0.5)
    s = e.snapshot()
    print(s['hand_visible'], s['max_angles'], s['grip_status'])
e.stop()