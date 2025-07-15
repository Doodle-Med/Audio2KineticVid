# Audio2KineticVid

Audio2KineticVid is a comprehensive tool that converts an audio track (e.g., a song) into a dynamic music video with AI-generated scenes and synchronized kinetic typography (animated subtitles). Everything runs locally using open-source models – no external APIs or paid services required.

## ✨ Features

- **🎤 Whisper Transcription:** Choose from multiple Whisper models (tiny to large) for audio transcription with word-level timestamps.
- **🧠 Adaptive Lyric Segmentation:** Splits lyrics into segments at natural pause points to align scene changes with the song.
- **🎨 Customizable Scene Generation:** Use various LLM models to generate scene descriptions for each lyric segment, with customizable system prompts and word limits.
- **🤖 Multiple AI Models:** Select from a variety of text-to-image models (SDXL, SD 1.5, etc.) and video generation models.
- **🎬 Style Consistency Options:** Choose between independent scene generation or img2img-based style consistency for a more cohesive visual experience.
- **🔍 Preview & Inspection:** Preview scenes before full generation and inspect all generated images in a gallery view.
- **🔄 Seamless Transitions:** Configurable crossfade transitions between scene clips.
- **🎪 Kinetic Subtitles:** PyCaps renders styled animated subtitles that appear in sync with the original audio.
- **🔒 Fully Local & Open-Source:** All models are open-license and run on local GPU.

## 💻 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended: RTX 3080/4070 or better)
- **RAM**: 16GB+ system RAM 
- **Storage**: SSD recommended for faster model loading and video processing
- **CPU**: Modern multi-core processor

### Software Requirements
- **Operating System**: Linux, Windows, or macOS
- **Python**: 3.8 or higher
- **CUDA**: NVIDIA CUDA toolkit (for GPU acceleration)
- **FFmpeg**: For audio/video processing

## 🚀 Quick Start (Gradio Web UI)

### 1. Install Dependencies

Ensure you have a suitable GPU (NVIDIA T4/A10 or better) with CUDA installed. Then install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Interface

```bash
python app.py
```

This will start a Gradio web interface accessible at `http://localhost:7860`.

### 3. Using the Interface

1. **Upload Audio**: Choose an audio file (MP3, WAV, M4A, etc.)
2. **Select Quality Preset**: Choose from Fast, Balanced, or High Quality
3. **Configure Models**: Optionally adjust AI models in the "AI Models" tab
4. **Customize Style**: Modify scene prompts and visual style in other tabs  
5. **Preview**: Click "Preview First Scene" to test settings quickly
6. **Generate**: Click "Generate Complete Music Video" to create the full video

## 📝 Usage Tips

### Audio Selection
- **Format**: MP3, WAV, M4A, FLAC, OGG supported
- **Quality**: Clear vocals work best for transcription
- **Length**: 30 seconds to 3 minutes recommended for testing
- **Content**: Songs with distinct lyrics produce better results

### Performance Optimization
- **Fast Generation**: Use 512x288 resolution with "tiny" Whisper model
- **Best Quality**: Use 1280x720 with "large" Whisper model (requires more VRAM)
- **Memory Issues**: Lower resolution, use smaller models, or reduce max segments

### Style Customization
- **Visual Style Keywords**: Add style terms like "cinematic, vibrant, neon" to influence all scenes
- **Prompt Template**: Customize how the AI interprets lyrics into visual scenes
- **Consistency Mode**: Use "Consistent (Img2Img)" for coherent visual style across scenes

## 🛠️ Advanced Usage

### Command Line Interface

For batch processing or automation, you can use the smoke test script:

```bash
bash scripts/smoke_test.sh your_audio.mp3
```

### Custom Templates

Create custom subtitle styles by adding new templates in the `templates/` directory:

1. Create a new folder: `templates/your_style/`
2. Add `pycaps.template.json` with animation definitions
3. Add `styles.css` with visual styling
4. The template will appear in the interface dropdown

### Model Configuration

Supported models are defined in the utility modules:
- **Whisper**: `utils/transcribe.py` - Add new Whisper model names
- **LLM**: `utils/prompt_gen.py` - Add new language models  
- **Image**: `utils/video_gen.py` - Add new Stable Diffusion variants
- **Video**: `utils/video_gen.py` - Add new video diffusion models

## 🧪 Testing

Run the basic functionality test:

```bash
python test_basic.py
```

For a complete end-to-end test with a sample audio file:

```bash
python test.py
```

## 📁 Project Structure

```
Audio2KineticVid/
├── app.py                  # Main Gradio web interface
├── requirements.txt        # Python dependencies
├── utils/                  # Core processing modules
│   ├── transcribe.py      # Whisper audio transcription
│   ├── segment.py         # Intelligent lyric segmentation  
│   ├── prompt_gen.py      # LLM scene description generation
│   ├── video_gen.py       # Image and video generation
│   └── glue.py           # Video stitching and subtitle overlay
├── templates/             # Subtitle animation templates
│   ├── minimalist/       # Clean, simple subtitle style
│   └── dynamic/          # Dynamic animations
├── scripts/              # Utility scripts
│   └── smoke_test.sh     # End-to-end testing script
└── test_basic.py         # Component testing
```

## 🎬 Output

The application generates:
- **Final Video**: MP4 file with synchronized audio, visuals, and animated subtitles
- **Scene Images**: Individual AI-generated images for each lyric segment
- **Scene Descriptions**: Text prompts used for image generation
- **Segmentation Data**: Analyzed lyric segments with timing information

## 🔧 Troubleshooting

### Common Issues

**GPU Memory Errors**
- Reduce video resolution (use 512x288 instead of 1280x720)
- Use smaller models (tiny/base Whisper, SD 1.5 instead of SDXL)
- Close other GPU-intensive applications

**Audio Processing Fails**
- Ensure FFmpeg is installed and accessible
- Try converting audio to WAV format first
- Check that audio file is not corrupted

**Model Loading Issues**
- Check internet connection (models download on first use)
- Verify sufficient disk space for model files
- Clear HuggingFace cache if models are corrupted

**Slow Generation**
- Use "Fast" quality preset for testing
- Reduce crossfade duration to 0 for hard cuts
- Use dynamic FPS instead of fixed high FPS

### Performance Monitoring

Monitor system resources during generation:
- **GPU Usage**: Should be near 100% during image/video generation
- **RAM Usage**: Peak during model loading and video processing
- **Disk I/O**: High during model downloads and video encoding

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional subtitle animation templates
- Support for more AI models
- Performance optimizations
- Additional audio/video formats
- Batch processing capabilities

## 📄 License

This project uses open-source models and libraries. Please check individual model licenses for usage rights.

## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Stability AI** for Stable Diffusion models  
- **Hugging Face** for model hosting and transformers
- **PyCaps** for kinetic subtitle rendering
- **Gradio** for the web interface