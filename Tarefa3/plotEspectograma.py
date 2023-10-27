import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file = 'Senoide5kRuidoIndustrial1.mp3'
duration = 30

y, sr = librosa.load(audio_file, duration=duration)

spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma')
plt.show()