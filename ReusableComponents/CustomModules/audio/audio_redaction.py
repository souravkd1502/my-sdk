import json
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine

# ----------------- Config -----------------
transcript_file = Path("transcript_debug.json")
audio_file = Path("C:\\Users\\OMEN\\Downloads\\personal information.mp3")
output_file = Path("call_recording_redacted.wav")

# Redacted transcript (with stars)
redacted_text = """Thank you. Okay, what's your name? ************** Oon is the surname, yeah? Yeah. 
How do you spell that? *****. And what's your first name? *********. *******. Uh-huh. *******. 
Okay, that's fine. And how old are you? I'm ******, okay. And what's your contact phone number? 
It's **********, yes. 3904. Yeah. 237. 237, okay. That's *************. That's it. 
And what's your address and postcode? It's ****************. 21A. ************. Yeah. *********. 
And what's your postcode? ******9-***. 9-***. Okay, that's fine. Thank you very much. Thank you. Goodbye. Bye.
"""

# ----------------- Step 1. Load Whisper transcript -----------------
with open(transcript_file, "r", encoding="utf-8") as f:
    transcript = json.load(f)

words = transcript["words"]
original_text = transcript["text"]

# ----------------- Step 2. Compare original vs redacted -----------------
# Strategy: align both texts character by character
redaction_intervals = []
idx = 0
for w in words:
    w_clean = w["word"].strip()
    start, end = w["start"], w["end"]

    # Find this word in original text starting from idx
    pos = original_text.find(w_clean, idx)
    if pos == -1:
        continue
    idx = pos + len(w_clean)

    # If the corresponding span in redacted_text contains '*', mark redaction
    redacted_fragment = redacted_text[pos:pos+len(w_clean)]
    if "*" in redacted_fragment:
        redaction_intervals.append((start, end))

# Merge consecutive redaction intervals
merged_intervals = []
for s, e in sorted(redaction_intervals):
    if not merged_intervals or s > merged_intervals[-1][1]:
        merged_intervals.append([s, e])
    else:
        merged_intervals[-1][1] = e

# Convert to ms
redaction_intervals_ms = [(int(s*1000), int(e*1000)) for s, e in merged_intervals]
print("Redaction intervals (ms):", redaction_intervals_ms)

# ----------------- Step 3. Apply redactions -----------------
audio = AudioSegment.from_file(audio_file)
redacted_audio = audio

for start, end in redaction_intervals_ms:
    duration = end - start
    replacement = Sine(1000).to_audio_segment(duration=duration).apply_gain(-5)
    redacted_audio = redacted_audio[:start] + replacement + redacted_audio[end:]

# ----------------- Step 4. Save result -----------------
redacted_audio.export(output_file, format="wav")
print(f"✅ Redacted audio saved at {output_file}")
