#!/usr/bin/env python3
import os
import shutil
import uuid
import json
import gradio as gr
import torch
from PIL import Image
import time

# Import pipeline modules
from utils.transcribe import transcribe_audio, list_available_whisper_models
from utils.segment import segment_lyrics
from utils.prompt_gen import generate_scene_prompts, list_available_llm_models
from utils.video_gen import (
    create_video_segments, 
    list_available_image_models,
    list_available_video_models,
    preview_image_generation
)
from utils.glue import stitch_and_caption

# Create output directories if not existing
os.makedirs("templates", exist_ok=True)
os.makedirs("templates/minimalist", exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# Load available model options
WHISPER_MODELS = list_available_whisper_models()
DEFAULT_WHISPER_MODEL = "medium.en"

LLM_MODELS = list_available_llm_models()
DEFAULT_LLM_MODEL = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"

IMAGE_MODELS = list_available_image_models()
DEFAULT_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

VIDEO_MODELS = list_available_video_models()
DEFAULT_VIDEO_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """You are a cinematographer generating a scene for a music video.
Describe one vivid visual scene ({max_words} words max) that matches the mood and imagery of these lyrics.
Focus on setting, atmosphere, lighting, and framing. Do not mention the artist or singing.
Use only {max_sentences} sentence(s).

Lyrics: "{lyrics}"

Scene description:"""

# Prepare style template options by scanning templates/ directory
TEMPLATE_DIR = "templates"
template_choices = []
for name in os.listdir(TEMPLATE_DIR):
    if os.path.isdir(os.path.join(TEMPLATE_DIR, name)):
        template_choices.append(name)
template_choices = sorted(template_choices)
DEFAULT_TEMPLATE = "minimalist" if "minimalist" in template_choices else (template_choices[0] if template_choices else None)

# Advanced settings defaults
DEFAULT_RESOLUTION = "1024x576"  # default resolution
DEFAULT_FPS_MODE = "Auto"       # auto-match lyric timing
DEFAULT_SEED = 0                # 0 means random seed
DEFAULT_MAX_WORDS = 30          # default word limit for scene descriptions
DEFAULT_MAX_SENTENCES = 1       # default sentence limit
DEFAULT_CROSSFADE = 0.25        # default crossfade duration
DEFAULT_STYLE_SUFFIX = "cinematic, 35 mm, shallow depth of field, film grain"

# Mode for image generation
IMAGE_MODES = ["Independent", "Consistent (Img2Img)"]
DEFAULT_IMAGE_MODE = "Independent"

def process_audio(
    audio_path, 
    whisper_model, 
    llm_model,
    image_model,
    video_model,
    template_name, 
    resolution, 
    fps_mode, 
    seed,
    prompt_template,
    max_words,
    max_sentences,
    style_suffix,
    image_mode,
    strength,
    crossfade_duration,
    progress=None
):
    """
    End-to-end processing function to generate the music video with kinetic subtitles.
    Returns final video file path for preview and download.
    """
    if progress is None:
        # Default progress function just prints to console
        progress = lambda percent, desc="": print(f"Progress: {percent}% - {desc}")
    
    # Input validation
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("Please provide a valid audio file")
    
    if not template_name or template_name not in template_choices:
        template_name = DEFAULT_TEMPLATE or "minimalist"
        
    # Prepare a unique temp directory for this run (to avoid conflicts between parallel jobs)
    session_id = str(uuid.uuid4())[:8]
    work_dir = os.path.join("tmp", f"run_{session_id}")
    os.makedirs(work_dir, exist_ok=True)
    
    # Save parameter settings for debugging
    params = {
        "whisper_model": whisper_model,
        "llm_model": llm_model,
        "image_model": image_model,
        "video_model": video_model,
        "template": template_name,
        "resolution": resolution,
        "fps_mode": fps_mode,
        "seed": seed,
        "max_words": max_words,
        "max_sentences": max_sentences,
        "style_suffix": style_suffix,
        "image_mode": image_mode,
        "strength": strength,
        "crossfade_duration": crossfade_duration
    }
    with open(os.path.join(work_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    
    try:
        # 1. Transcription
        progress(0, desc="Transcribing audio with Whisper...")
        try:
            result = transcribe_audio(audio_path, whisper_model)
            if not result or 'segments' not in result:
                raise ValueError("Transcription failed - no speech detected")
        except Exception as e:
            raise RuntimeError(f"Audio transcription failed: {str(e)}")
        
        progress(15, desc="Transcription completed. Segmenting lyrics...")
        
        # 2. Segmentation
        try:
            segments = segment_lyrics(result)
            if not segments:
                raise ValueError("No valid segments found in transcription")
        except Exception as e:
            raise RuntimeError(f"Audio segmentation failed: {str(e)}")
        
        progress(25, desc=f"Detected {len(segments)} lyric segments. Generating scene prompts...")
        
        # 3. Scene-prompt generation
        try:
            # Format the prompt template with the limits
            formatted_prompt_template = prompt_template.format(
                max_words=max_words,
                max_sentences=max_sentences,
                lyrics="{lyrics}"  # This placeholder will be filled for each segment
            )
            
            prompts = generate_scene_prompts(
                segments, 
                llm_model=llm_model,
                prompt_template=formatted_prompt_template,
                style_suffix=style_suffix
            )
            
            if len(prompts) != len(segments):
                raise ValueError(f"Prompt generation mismatch: {len(prompts)} prompts for {len(segments)} segments")
                
        except Exception as e:
            raise RuntimeError(f"Scene prompt generation failed: {str(e)}")
        
        # Save generated prompts for display or debugging
        with open(os.path.join(work_dir, "prompts.txt"), "w", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                f.write(f"Segment {i+1}: {p}\n")
        progress(35, desc="Scene prompts ready. Generating video segments...")
        
        # Parse resolution with validation
        try:
            if resolution and "x" in resolution.lower():
                width, height = map(int, resolution.lower().split("x"))
                if width <= 0 or height <= 0:
                    raise ValueError("Invalid resolution values")
            else:
                width, height = 1024, 576  # default high resolution
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid resolution '{resolution}', using default 1024x576")
            width, height = 1024, 576
        
        # Determine FPS handling
        fps_value = None
        dynamic_fps = True
        if fps_mode and fps_mode.lower() != "auto":
            try:
                fps_value = float(fps_mode)
                if fps_value <= 0:
                    raise ValueError("FPS must be positive")
                dynamic_fps = False
            except (ValueError, TypeError):
                print(f"Warning: Invalid FPS '{fps_mode}', using auto mode")
                fps_value = None
                dynamic_fps = True
        
        # 4. Image→video generation for each segment
        try:
            segment_videos = create_video_segments(
                segments, 
                prompts, 
                image_model=image_model,
                video_model=video_model,
                width=width, 
                height=height,
                dynamic_fps=dynamic_fps, 
                base_fps=fps_value, 
                seed=seed, 
                work_dir=work_dir,
                image_mode=image_mode,
                strength=strength,
                progress_callback=lambda percent, desc: progress(35 + int(percent * 0.45), desc)
            )
            
            if not segment_videos:
                raise ValueError("No video segments were generated")
                
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
        
        progress(80, desc="Video segments generated. Stitching and adding subtitles...")
        
        # 5. Concatenation & audio syncing, plus kinetic subtitles overlay
        try:
            final_video_path = stitch_and_caption(
                segment_videos, 
                audio_path, 
                segments, 
                template_name, 
                work_dir=work_dir,
                crossfade_duration=crossfade_duration
            )
            
            if not final_video_path or not os.path.exists(final_video_path):
                raise ValueError("Final video file was not created")
                
        except Exception as e:
            raise RuntimeError(f"Video stitching and captioning failed: {str(e)}")
        
        progress(100, desc="✅ Generation complete!")
        return final_video_path, work_dir
        
    except Exception as e:
        # Enhanced error reporting
        error_msg = str(e)
        if "CUDA" in error_msg or "GPU" in error_msg:
            error_msg += "\n\n💡 Tip: This application requires a CUDA-compatible GPU with sufficient VRAM."
        elif "model" in error_msg.lower():
            error_msg += "\n\n💡 Tip: Model loading failed. Check your internet connection and try again."
        elif "audio" in error_msg.lower():
            error_msg += "\n\n💡 Tip: Please ensure your audio file is in a supported format (MP3, WAV, M4A)."
        
        print(f"Error during processing: {error_msg}")
        raise RuntimeError(error_msg)

# Define Gradio UI components
with gr.Blocks(title="Audio → Kinetic-Subtitle Music Video", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 Audio → Kinetic-Subtitle Music Video
    
    Transform your audio tracks into dynamic music videos with AI-generated scenes and animated subtitles.
    
    **✨ Features:**
    - 🎤 **Whisper Transcription** - Accurate speech-to-text with word-level timing
    - 🧠 **AI Scene Generation** - LLM-powered visual descriptions from lyrics
    - 🎨 **Image & Video AI** - Stable Diffusion + Video Diffusion models
    - 🎬 **Kinetic Subtitles** - Animated text synchronized with audio
    - ⚡ **Fully Local** - No API keys required, runs on your GPU
    
    **📋 Quick Start:**
    1. Upload an audio file (MP3, WAV, M4A)
    2. Choose your AI models (or keep defaults)
    3. Customize style and settings
    4. Click "Generate Music Video"
    """)
    
    # System requirements info
    with gr.Accordion("💻 System Requirements & Tips", open=False):
        gr.Markdown("""
        **Hardware Requirements:**
        - NVIDIA GPU with 8GB+ VRAM (recommended: RTX 3080/4070 or better)
        - 16GB+ system RAM
        - Fast storage (SSD recommended)
        
        **Supported Audio Formats:**
        - MP3, WAV, M4A, FLAC, OGG
        - Recommended: Clear vocals, 30 seconds to 3 minutes
        
        **Performance Tips:**
        - Use lower resolution (512x288) for faster generation
        - Choose smaller models for quicker processing
        - Ensure stable power supply for GPU-intensive tasks
        """)
    
    # Main configuration
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="🎵 Upload Audio Track", 
                type="filepath",
                info="Upload your music file. For best results, use clear audio with distinct vocals."
            )
        with gr.Column():
            # Quick settings panel
            gr.Markdown("### ⚡ Quick Settings")
            quick_quality = gr.Radio(
                choices=["Fast (512x288)", "Balanced (1024x576)", "High Quality (1280x720)"],
                value="Balanced (1024x576)",
                label="Quality Preset",
                info="Higher quality = better results but slower generation"
            )
    
    # Model selection tabs
    with gr.Tabs():
        with gr.TabItem("🤖 AI Models"):
            gr.Markdown("**Choose the AI models for each processing step:**")
            with gr.Row():
                with gr.Column():
                    whisper_dropdown = gr.Dropdown(
                        label="🎤 Transcription Model (Whisper)", 
                        choices=WHISPER_MODELS, 
                        value=DEFAULT_WHISPER_MODEL,
                        info="Larger models are more accurate but slower. 'medium.en' is recommended for English."
                    )
                    llm_dropdown = gr.Dropdown(
                        label="🧠 Scene Description Model (LLM)",
                        choices=LLM_MODELS,
                        value=DEFAULT_LLM_MODEL,
                        info="Language model to generate visual scene descriptions from lyrics."
                    )
                with gr.Column():
                    image_dropdown = gr.Dropdown(
                        label="🎨 Image Generation Model",
                        choices=IMAGE_MODELS,
                        value=DEFAULT_IMAGE_MODEL,
                        info="Stable Diffusion model for generating scene images."
                    )
                    video_dropdown = gr.Dropdown(
                        label="🎬 Video Animation Model",
                        choices=VIDEO_MODELS,
                        value=DEFAULT_VIDEO_MODEL,
                        info="Model to animate still images into video clips."
                    )
        
        with gr.TabItem("✍️ Scene Prompting"):
            gr.Markdown("**Customize how AI generates scene descriptions:**")
            with gr.Column():
                prompt_template_input = gr.Textbox(
                    label="LLM Prompt Template",
                    value=DEFAULT_PROMPT_TEMPLATE,
                    lines=6,
                    info="Template for generating scene descriptions. Use {lyrics}, {max_words}, and {max_sentences} as placeholders."
                )
                with gr.Row():
                    max_words_input = gr.Slider(
                        label="Max Words per Scene",
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=DEFAULT_MAX_WORDS,
                        info="Limit words in each scene description (more words = more detailed scenes)."
                    )
                    max_sentences_input = gr.Slider(
                        label="Max Sentences per Scene",
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=DEFAULT_MAX_SENTENCES,
                        info="Limit sentences per scene (1-2 recommended for music videos)."
                    )
                style_suffix_input = gr.Textbox(
                    label="Visual Style Keywords",
                    value=DEFAULT_STYLE_SUFFIX,
                    info="Style keywords added to all scenes for consistent visual style (e.g., 'cinematic, vibrant colors')."
                )
        
        with gr.TabItem("🎬 Video Settings"):
            gr.Markdown("**Configure video output and subtitle styling:**")
            with gr.Column():
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        label="🎪 Subtitle Animation Style",
                        choices=template_choices,
                        value=DEFAULT_TEMPLATE,
                        info="Choose the kinetic subtitle animation style."
                    )
                    res_dropdown = gr.Dropdown(
                        label="📺 Video Resolution",
                        choices=["512x288", "1024x576", "1280x720"],
                        value=DEFAULT_RESOLUTION,
                        info="Higher resolution = better quality but much slower generation."
                    )
                with gr.Row():
                    fps_input = gr.Textbox(
                        label="🎞️ Video FPS",
                        value=DEFAULT_FPS_MODE,
                        info="Frames per second. Use 'Auto' to match lyric timing, or set fixed value (e.g., '24', '30')."
                    )
                    seed_input = gr.Number(
                        label="🌱 Random Seed",
                        value=DEFAULT_SEED,
                        precision=0,
                        info="Set seed for reproducible results (0 = random). Use same seed to recreate results."
                    )
                with gr.Row():
                    image_mode_input = gr.Radio(
                        label="🖼️ Scene Generation Mode",
                        choices=IMAGE_MODES,
                        value=DEFAULT_IMAGE_MODE,
                        info="Independent: each scene is unique. Consistent: scenes influence each other for style continuity."
                    )
                    strength_slider = gr.Slider(
                        label="🎯 Style Consistency Strength",
                        minimum=0.1,
                        maximum=0.9,
                        step=0.05,
                        value=0.5,
                        visible=False,
                        info="How much each scene influences the next (lower = more influence, higher = more variety)."
                    )
                crossfade_slider = gr.Slider(
                    label="🔄 Scene Transition Duration",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=DEFAULT_CROSSFADE,
                    info="Smooth crossfade between scenes in seconds (0 = hard cuts, 0.25 = subtle blend)."
                )
    
    # Quick preset handling
    def apply_quality_preset(preset):
        if preset == "Fast (512x288)":
            return gr.update(value="512x288"), gr.update(value="tiny"), gr.update(value="stabilityai/sdxl-turbo")
        elif preset == "High Quality (1280x720)":
            return gr.update(value="1280x720"), gr.update(value="large"), gr.update(value="stabilityai/stable-diffusion-xl-base-1.0")
        else:  # Balanced
            return gr.update(value="1024x576"), gr.update(value="medium.en"), gr.update(value="stabilityai/stable-diffusion-xl-base-1.0")
    
    quick_quality.change(
        apply_quality_preset,
        inputs=[quick_quality],
        outputs=[res_dropdown, whisper_dropdown, image_dropdown]
    )
    
    # Make strength slider visible only when Consistent mode is selected
    def update_strength_visibility(mode):
        return gr.update(visible=(mode == "Consistent (Img2Img)"))
    
    image_mode_input.change(update_strength_visibility, inputs=image_mode_input, outputs=strength_slider)

    # Enhanced preview section
    with gr.Row():
        with gr.Column(scale=1):
            preview_btn = gr.Button("🔍 Preview First Scene", variant="secondary", size="lg")
            gr.Markdown("*Generate a quick preview of the first scene to test your settings*")
        with gr.Column(scale=2):
            generate_btn = gr.Button("🎬 Generate Complete Music Video", variant="primary", size="lg")
            gr.Markdown("*Start the full video generation process (this may take several minutes)*")
    
    # Preview results
    with gr.Row(visible=False) as preview_row:
        with gr.Column():
            preview_img = gr.Image(label="Preview Scene", type="pil", height=300)
        with gr.Column():
            preview_prompt = gr.Textbox(label="Generated Scene Description", lines=3)
            preview_info = gr.Markdown("")
    
    # Progress and status
    progress_bar = gr.Progress()
    status_text = gr.Textbox(
        label="📊 Generation Status", 
        value="Ready to start...",
        interactive=False,
        lines=2
    )
    
    # Results section with better organization
    with gr.Tabs():
        with gr.TabItem("🎥 Final Video"):
            output_video = gr.Video(label="Generated Music Video", format="mp4", height=400)
            with gr.Row():
                download_file = gr.File(label="📥 Download Video File", file_count="single")
                video_info = gr.Textbox(label="Video Information", lines=2, interactive=False)
        
        with gr.TabItem("🖼️ Generated Images"):
            image_gallery = gr.Gallery(
                label="Scene Images from Video Generation",
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain",
                show_label=True
            )
            gallery_info = gr.Markdown("*Scene images will appear here after generation*")
        
        with gr.TabItem("📝 Scene Descriptions"):
            with gr.Accordion("Generated Scene Prompts", open=True):
                prompt_text = gr.Markdown("", elem_id="prompt_markdown")
            segment_info = gr.Textbox(
                label="Segmentation Summary", 
                lines=3, 
                interactive=False,
                placeholder="Segment analysis will appear here..."
            )
    
    # Preview function
    def on_preview(
        audio, whisper_model, llm_model, image_model, 
        prompt_template, max_words, max_sentences, style_suffix, resolution
    ):
        if not audio:
            return (gr.update(visible=False), None, "Please upload audio first", 
                   "⚠️ **No audio file provided**\n\nPlease upload an audio file to generate a preview.")
        
        # Quick transcription and segmentation of first few seconds
        try:
            # Extract first 10 seconds of audio for quick preview
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Use ffmpeg to extract first 10 seconds
            subprocess.run([
                "ffmpeg", "-y", "-i", audio, "-ss", "0", "-t", "10", 
                "-acodec", "pcm_s16le", temp_audio_path
            ], check=True, capture_output=True, stderr=subprocess.DEVNULL)
            
            # Transcribe with fastest model for preview
            result = transcribe_audio(temp_audio_path, "tiny")
            segments = segment_lyrics(result)
            os.unlink(temp_audio_path)
            
            if not segments:
                return (gr.update(visible=False), None, "No speech detected in first 10 seconds", 
                       "⚠️ **No speech detected**\n\nTry with audio that has clear vocals at the beginning.")
            
            first_segment = segments[0]
            
            # Format prompt template
            formatted_prompt = prompt_template.format(
                max_words=max_words,
                max_sentences=max_sentences,
                lyrics=first_segment["text"]
            )
            
            # Generate prompt
            scene_prompt = generate_scene_prompts(
                [first_segment], 
                llm_model=llm_model,
                prompt_template=formatted_prompt,
                style_suffix=style_suffix
            )[0]
            
            # Generate image
            if resolution and "x" in resolution.lower():
                width, height = map(int, resolution.lower().split("x"))
            else:
                width, height = 1024, 576
                
            image = preview_image_generation(
                scene_prompt, 
                image_model=image_model,
                width=width,
                height=height
            )
            
            # Create info text
            duration = first_segment['end'] - first_segment['start']
            info_text = f"""
✅ **Preview Generated Successfully**

**Detected Lyrics:** "{first_segment['text'][:100]}{'...' if len(first_segment['text']) > 100 else ''}"

**Scene Duration:** {duration:.1f} seconds

**Generated Description:** {scene_prompt[:150]}{'...' if len(scene_prompt) > 150 else ''}

**Image Resolution:** {width}x{height}
            """
            
            return gr.update(visible=True), image, scene_prompt, info_text
            
        except subprocess.CalledProcessError as e:
            return (gr.update(visible=False), None, "Audio processing failed", 
                   "❌ **Audio Processing Error**\n\nFFmpeg failed to process the audio file. Please check the format.")
        except Exception as e:
            print(f"Preview error: {e}")
            return (gr.update(visible=False), None, f"Preview failed: {str(e)}", 
                   f"❌ **Preview Error**\n\n{str(e)}\n\nPlease check your audio file and model settings.")

    # Bind button click to processing function
    def on_generate(
        audio, whisper_model, llm_model, image_model, video_model,
        template_name, resolution, fps, seed, prompt_template,
        max_words, max_sentences, style_suffix, image_mode, strength,
        crossfade_duration, progress=gr.Progress()
    ):
        if not audio:
            return (None, None, gr.update(value="**No audio file provided**\n\nPlease upload an audio file to start generation.", visible=True), 
                   [], "Ready to start...", "", "")
            
        try:
            # Enhanced progress callback function
            def update_progress(percent, desc=""):
                progress(percent / 100, desc)
                return f"🔄 **Generation in Progress:** {percent:.0f}%\n\n{desc}"
            
            # Start generation
            start_time = time.time()
            final_path, work_dir = process_audio(
                audio, whisper_model, llm_model, image_model, video_model,
                template_name, resolution, fps, int(seed), prompt_template,
                max_words, max_sentences, style_suffix, image_mode, strength,
                crossfade_duration, progress=update_progress
            )
            
            generation_time = time.time() - start_time
            
            # Load prompts from file to display
            prompts_file = os.path.join(work_dir, "prompts.txt")
            prompts_markdown = ""
            try:
                with open(prompts_file, 'r', encoding='utf-8') as pf:
                    content = pf.read()
                    # Format prompts as numbered list
                    prompts_lines = content.strip().splitlines()
                    prompts_markdown = "\n".join([f"**{line}**" for line in prompts_lines])
            except:
                prompts_markdown = "Scene prompts not available"
            
            # Load segment information
            segment_summary = ""
            try:
                # Get audio duration and file info
                import subprocess
                duration_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                              "-of", "default=noprint_wrappers=1:nokey=1", audio]
                audio_duration = float(subprocess.check_output(duration_cmd, text=True).strip())
                
                file_size = os.path.getsize(final_path) / (1024 * 1024)  # MB
                segment_summary = f"""📊 **Generation Summary:**
• Audio Duration: {audio_duration:.1f} seconds
• Processing Time: {generation_time/60:.1f} minutes  
• Final Video Size: {file_size:.1f} MB
• Resolution: {resolution}
• Template: {template_name}"""
            except:
                segment_summary = f"Generation completed in {generation_time/60:.1f} minutes"
                
            # Load generated images for the gallery
            images = []
            try:
                import glob
                image_files = glob.glob(os.path.join(work_dir, "*_img.png"))
                for img_file in sorted(image_files):
                    try:
                        img = Image.open(img_file)
                        images.append(img)
                    except:
                        pass
            except Exception as e:
                print(f"Error loading images for gallery: {e}")
            
            # Create video info
            video_info = f"✅ Video generated successfully!\nFile: {os.path.basename(final_path)}\nSize: {file_size:.1f} MB"
            gallery_info_text = f"**{len(images)} scene images generated**" if images else "No images available"
            
            return (final_path, final_path, gr.update(value=prompts_markdown, visible=True), 
                   images, f"✅ Generation complete! ({generation_time/60:.1f} minutes)", 
                   video_info, segment_summary)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            
            return (None, None, gr.update(value=f"**❌ Generation Failed**\n\n{error_msg}", visible=True), 
                   [], f"❌ Error: {error_msg}", "", "")

    preview_btn.click(
        on_preview,
        inputs=[
            audio_input, whisper_dropdown, llm_dropdown, image_dropdown, 
            prompt_template_input, max_words_input, max_sentences_input,
            style_suffix_input, res_dropdown
        ],
        outputs=[preview_row, preview_img, preview_prompt, preview_info]
    )

    generate_btn.click(
        on_generate,
        inputs=[
            audio_input, whisper_dropdown, llm_dropdown, image_dropdown, video_dropdown,
            template_dropdown, res_dropdown, fps_input, seed_input, prompt_template_input,
            max_words_input, max_sentences_input, style_suffix_input,
            image_mode_input, strength_slider, crossfade_slider
        ],
        outputs=[output_video, download_file, prompt_text, image_gallery, status_text, video_info, segment_info]
    )

if __name__ == "__main__":
    # Uncomment for custom hosting options
    # demo.launch(server_name='0.0.0.0', server_port=7860)
    demo.launch()