import whisper

# Cache loaded whisper models to avoid reloading for each request
_model_cache = {}

def list_available_whisper_models():
    """Return list of available Whisper models"""
    return ["tiny", "base", "small", "medium", "medium.en", "large", "large-v2"]

def transcribe_audio(audio_path: str, model_size: str = "medium.en"):
    """
    Transcribe the given audio file using OpenAI Whisper and return the result dictionary.
    The result includes per-word timestamps.
    
    Args:
        audio_path: Path to the audio file
        model_size: Size of Whisper model to use (tiny, base, small, medium, medium.en, large)
        
    Returns:
        Dictionary with transcription results including segments with word timestamps
    """
    # Load model (use cache if available)
    model_size = model_size or "medium.en"
    if model_size not in _model_cache:
        # Load Whisper model
        print(f"Loading Whisper model: {model_size}...")
        _model_cache[model_size] = whisper.load_model(model_size)
    model = _model_cache[model_size]
    # Perform transcription with word-level timestamps
    result = model.transcribe(audio_path, word_timestamps=True, verbose=False, task="transcribe", language="en")
    # The result is a dict with "text" and "segments". Each segment may include 'words' list for word-level timestamps.
    return result