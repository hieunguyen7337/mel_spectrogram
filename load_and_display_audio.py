import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_file = "test.wav"
y, sr = librosa.load(audio_file, sr=16000, mono=True)

print('y:', y, '\n')
print('y shape:', np.shape(y), '\n')
print('Sample rate (KHz):', sr, '\n')
print(f'Length of audio: {np.shape(y)[0]/sr}')

plt.figure(figsize=(15, 5))
librosa.display.waveshow(y=y, sr=sr);
plt.title("Sound wave of test.wav", fontsize=20)
plt.show()