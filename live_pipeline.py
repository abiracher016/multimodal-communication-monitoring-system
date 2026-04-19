import time
import whisper
from textblob import TextBlob
import pandas as pd
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write

model = whisper.load_model("base")

fs = 16000
seconds = 5

print("🎥 Live Monitoring Started... Press CTRL+C to stop")

while True:
    print("\n🎤 Recording chunk...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()

    write("live_audio.wav", fs, audio)

    result = model.transcribe("live_audio.wav")
    text = result["text"]

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    speaker = "Interviewer" if len(text) < 60 else "Candidate"

    new_data = pd.DataFrame([{
        "timestamp": datetime.now(),
        "text": text,
        "sentiment_score": polarity,
        "sentiment": sentiment,
        "speaker": speaker
    }])

    try:
        old = pd.read_csv("session_analysis.csv")
        updated = pd.concat([old, new_data])
    except:
        updated = new_data

    updated.to_csv("session_analysis.csv", index=False)

    print(f"📝 {text}")
    print(f"📊 Sentiment: {sentiment} ({polarity})")

    time.sleep(2)