import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000
seconds = 10

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("live_audio.wav", fs, audio)
print("Saved as live_audio.wav")