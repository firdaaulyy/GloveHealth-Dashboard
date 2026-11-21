import json
import os
import time
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from kalibrasi_engine import RECOMMENDATION, CalibrationEngine

app = FastAPI(title='GLOVE-HEALTH Doctor Dashboard')
engine = CalibrationEngine()
PROFILE_FILE = 'patient_profile.json'
ROM_MAP = {'Easy': 0.65, 'Medium': 0.85, 'Hard': 1.00}

class KalibrasiIn(BaseModel):
    patient_name: str = Field(min_length=1, max_length=80)
    patient_id: str = Field(default='', max_length=30, pattern=r'^[A-Za-z0-9_-]*$')
    affected_side: Literal['kiri', 'kanan']

class ResepIn(BaseModel):
    duration: int = Field(ge=3, le=60)
    difficulty: Literal['Easy', 'Medium', 'Hard']
    max_bpm: int = Field(ge=80, le=160)
    haptic: bool

def load_profile():
    if not os.path.exists(PROFILE_FILE):
        return None
    with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_profile(data, path=PROFILE_FILE):
    # Tulis ke berkas sementara lalu ganti, agar pembaca tidak menerima berkas setengah jadi
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

@app.get('/')
def index():
    return FileResponse('index.html')

@app.post('/api/kamera/mulai')
def kamera_mulai():
    engine.start()
    return {'ok': True}

@app.post('/api/kamera/berhenti')
def kamera_berhenti():
    engine.stop()
    return {'ok': True}

@app.post('/api/kalibrasi/reset')
def kalibrasi_reset():
    engine.reset()
    return {'ok': True}

@app.get('/api/kalibrasi/status')
def kalibrasi_status():
    return engine.snapshot()

@app.get('/video_feed')
def video_feed():
    def stream():
        while engine.running:
            frame = engine.get_frame()
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            time.sleep(0.04)
    return StreamingResponse(stream(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/api/kalibrasi/simpan')
def kalibrasi_simpan(body: KalibrasiIn):
    snap = engine.snapshot()
    if snap['valid_frames'] < 60 or snap['avg_max'] < 5.0:
        raise HTTPException(400, 'Data belum cukup. Minta pasien membuka dan mengepalkan tangan dengan tangan terlihat jelas.')
    now = datetime.now()
    durasi, level, haptic = RECOMMENDATION[snap['grip_status']]
    pid = body.patient_id or f'PSN-{now:%H%M%S}'
    profile = {
        'patient_id': pid,
        'patient_name': body.patient_name,
        'affected_side': body.affected_side,
        'calibration_timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
        'calibration_cv': {
            'grip_status': snap['grip_status'],
            'average_max_flexion': snap['avg_max'],
            'clinical_note': snap['clinical_note'],
            'max_flexion_angles': snap['max_angles'],
        },
        'movement_analysis': snap['movement'],
        'doctor_prescription': {
            'status': 'Menunggu Konfigurasi Dokter',
            'session_duration_minutes': durasi,
            'game_difficulty_level': level,
            'target_rom_percentage': ROM_MAP[level],
            'max_safe_bpm': 115,
            'enable_haptic_assist': haptic,
        },
    }
    save_profile(profile)
    os.makedirs('data/riwayat', exist_ok=True)
    save_profile(profile, f'data/riwayat/{pid}_{now:%Y%m%d_%H%M%S}.json')
    return profile

@app.get('/api/profil')
def get_profil():
    data = load_profile()
    if data is None:
        raise HTTPException(404, 'Belum ada kalibrasi tersimpan.')
    return data

@app.post('/api/resep')
def simpan_resep(body: ResepIn):
    data = load_profile()
    if data is None:
        raise HTTPException(409, 'Lakukan dan simpan kalibrasi terlebih dahulu.')
    data['doctor_prescription'] = {
        'status': 'Ditetapkan oleh Dokter',
        'session_duration_minutes': body.duration,
        'game_difficulty_level': body.difficulty,
        'target_rom_percentage': ROM_MAP[body.difficulty],
        'max_safe_bpm': body.max_bpm,
        'enable_haptic_assist': body.haptic,
        'prescribed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_profile(data)
    return data