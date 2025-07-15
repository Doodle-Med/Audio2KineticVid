# Audio2KineticVid

Audio2KineticVid is a comprehensive tool that converts an audio track (e.g., a song) into a dynamic music video with AI-generated scenes and synchronized kinetic typography (animated subtitles). Everything runs locally using open-source models – no external APIs or paid services required.

## Features

- **Whisper Transcription:** Choose from multiple Whisper models (tiny to large) for audio transcription with word-level timestamps.
- **Adaptive Lyric Segmentation:** Splits lyrics into segments at natural pause points to align scene changes with the song.
- **Customizable Scene Generation:** Use various LLM models to generate scene descriptions for each lyric segment, with customizable system prompts and word limits.
- **Multiple AI Models:** Select from a variety of text-to-image models (SDXL, SD 1.5, etc.) and video generation models.
- **Style Consistency Options:** Choose between independent scene generation or img2img-based style consistency for a more cohesive visual experience.
- **Preview & Inspection:** Preview scenes before full generation and inspect all generated images in a gallery view.
- **Seamless Transitions:** Configurable crossfade transitions between scene clips.
- **Kinetic Subtitles:** PyCaps renders styled animated subtitles that appear in sync with the original audio.
- **Fully Local & Open-Source:** All models are open-license and run on local GPU.

## Quick Start (Gradio Web UI)

1. **Install Dependencies:** Ensure you have a suitable GPU (NVIDIA T4/A10 or better) with CUDA installed. Then install the required Python packages:

   ```bash
   pip install -r requirements.txt