import sys
try:
    import pythoncom
    pythoncom.CoInitialize()
    import win32com.client
    voice = win32com.client.Dispatch("SAPI.SpVoice")

    # Mode: list voices or speak
    if len(sys.argv) > 1 and sys.argv[1] == "__list__":
        voices = voice.GetVoices()
        for i in range(voices.Count):
            print(f"{i}|{voices.Item(i).GetDescription()}")
    else:
        text = sys.argv[1] if len(sys.argv) > 1 else ""
        voice_idx = int(sys.argv[2]) if len(sys.argv) > 2 else -1
        rate = int(sys.argv[3]) if len(sys.argv) > 3 else 2

        if voice_idx >= 0:
            voices = voice.GetVoices()
            if voice_idx < voices.Count:
                voice.Voice = voices.Item(voice_idx)

        voice.Rate = rate
        voice.Volume = 100
        voice.Speak(text)

    pythoncom.CoUninitialize()
except Exception as e:
    sys.stderr.write(str(e))
