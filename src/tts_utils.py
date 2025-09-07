import os
import tempfile
from pathlib import Path


def speak_text(text: str, engine: str = 'pyttsx3') -> Path:
    """
    Speak text using the specified TTS engine.
    engine: 'pyttsx3' (offline, default) or 'gtts' (online)
    Returns path to generated audio file if created, else None.
    """
    engine = (engine or 'pyttsx3').lower()

    if engine == 'gtts':
        try:
            from gtts import gTTS
            tmp_dir = Path(tempfile.gettempdir())
            out_path = tmp_dir / "caption_tts.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(str(out_path))
            try:
                # Attempt to play automatically (best-effort, cross-platform)
                if os.name == 'nt':
                    os.startfile(str(out_path))  # Windows
                else:
                    # macOS 'afplay', many Linux 'xdg-open' or 'mpg123'
                    os.system(f"xdg-open '{out_path}' 2>/dev/null || afplay '{out_path}' 2>/dev/null || mpg123 '{out_path}' 2>/dev/null || true")
            except Exception:
                pass
            return out_path
        except Exception:
            # Fallback to pyttsx3 silently
            engine = 'pyttsx3'

    # Default/offline engine
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_engine.say(text)
        tts_engine.runAndWait()
        return None
    except Exception:
        return None



