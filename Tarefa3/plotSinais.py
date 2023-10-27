import librosa
import numpy as np
import matplotlib.pyplot as plt

# Visualização do áudio
audio_file = 'audios/Sprint 2/Senoide5kRuidoIndustrial1.mp3'

duration = 30

# 30 primeiros segundos do áudio
y, sr = librosa.load(audio_file, duration=duration)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, len(y)) / sr, y)
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.title('Forma de Onda dos Primeiros 30s do Áudio')

#Espectograma
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma')


# FFT
fft_result = np.fft.fft(y)

frequencies = np.fft.fftfreq(len(fft_result), 1.0 / sr)

plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
plt.title('Espectro de Frequência (FFT)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()






