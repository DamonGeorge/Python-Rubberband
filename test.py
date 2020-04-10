import numpy as np
from rubber import Rubber
import soundfile as sf
import sounddevice as sd


audio, sr = sf.read("/Users/damongeorge/Music/Canteloupe Island.wav", dtype="float32")
print(f"audio shape: {audio.shape}")
audio = audio[:2000000]

r = Rubber(sample_rate=sr, channels=audio.shape[1], realtime=True)
stretched = r.stretch(audio, 1.4344)
stretched.shape
