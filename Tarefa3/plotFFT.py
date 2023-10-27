import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file = 'Senoide5kRuidoIndustrial1.mp3'
duration = 30

y, sr = librosa.load(audio_file, duration=duration)

fft_result = np.fft.fft(y)

frequencies = np.fft.fftfreq(len(fft_result), 1.0 / sr)

plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])
plt.title('Espectro de Frequência (FFT)')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()