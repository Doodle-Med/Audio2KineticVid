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
        result = transcribe_audio(audio_path, whisper_model)
        progress(15, desc="Transcription completed. Segmenting lyrics...")
        
        # 2. Segmentation
        segments = segment_lyrics(result)
        progress(25, desc=f"Detected {len(segments)} lyric segments. Generating scene prompts...")
        
        # 3. Scene-prompt generation
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
        # Save generated prompts for display or debugging
        with open(os.path.join(work_dir, "prompts.txt"), "w", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                f.write(f"Segment {i+1}: {p}\n")
        progress(35, desc="Scene prompts ready. Generating video segments...")
        
        # Parse resolution
        if resolution and "x" in resolution.lower():
            width, height = map(int, resolution.lower().split("x"))
        else:
            width, height = 1024, 576  # default high resolution
        
        # Determine FPS handling
        fps_value = None
        dynamic_fps = True
        if fps_mode and fps_mode.lower() != "auto":
            try:
                fps_value = float(fps_mode)
                dynamic_fps = False
            except:
                fps_value = None
                dynamic_fps = True
        
        # 4. Image→video generation for each segment
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
        progress(80, desc="Video segments generated. Stitching and adding subtitles...")
        
        # 5. Concatenation & audio syncing, plus kinetic subtitles overlay
        final_video_path = stitch_and_caption(
            segment_videos, 
            audio_path, 
            segments, 
            template_name, 
            work_dir=work_dir,
            crossfade_duration=crossfade_duration
        )
        progress(100, desc="Done")
        return final_video_path, work_dir
    except Exception as e:
        print("Error during processing:", e)
        raise

# Define Gradio UI components
with gr.Blocks(title="Audio → Kinetic-Subtitle Music Video") as demo:
    gr.Markdown("## 🎵 Audio → Kinetic-Subtitle Music Video\nUpload an audio track and generate a music video with AI-generated scenes and animated subtitles.")
    
    # Main configuration
    with gr.Row():
        audio_input = gr.Audio(label="1. Upload audio track", type="filepath")
    
    # Model selection tabs
    with gr.Tabs():
        with gr.TabItem("Models"):
            with gr.Row():
                with gr.Column():
                    whisper_dropdown = gr.Dropdown(
                        label="Transcription Model (Whisper)", 
                        choices=WHISPER_MODELS, 
                        value=DEFAULT_WHISPER_MODEL,
                        info="Larger models are slower but more accurate for transcription."
                    )
                    llm_dropdown = gr.Dropdown(
                        label="Prompt Generation Model (LLM)",
                        choices=LLM_MODELS,
                        value=DEFAULT_LLM_MODEL,
                        info="Language model to generate scene descriptions from lyrics."
                    )
                with gr.Column():
                    image_dropdown = gr.Dropdown(
                        label="Image Generation Model",
                        choices=IMAGE_MODELS,
                        value=DEFAULT_IMAGE_MODEL,
                        info="Model used to generate still images from scene prompts."
                    )
                    video_dropdown = gr.Dropdown(
                        label="Video Generation Model",
                        choices=VIDEO_MODELS,
                        value=DEFAULT_VIDEO_MODEL,
                        info="Model used to animate still images into short video clips."
                    )
        
        with gr.TabItem("Scene Prompting"):
            with gr.Column():
                prompt_template_input = gr.Textbox(
                    label="LLM Prompt Template",
                    value=DEFAULT_PROMPT_TEMPLATE,
                    lines=6,
                    info="Template used for generating scene descriptions. Use {lyrics}, {max_words}, and {max_sentences} as placeholders."
                )
                with gr.Row():
                    max_words_input = gr.Slider(
                        label="Max Words per Scene",
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=DEFAULT_MAX_WORDS,
                        info="Maximum words per scene description."
                    )
                    max_sentences_input = gr.Slider(
                        label="Max Sentences per Scene",
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=DEFAULT_MAX_SENTENCES,
                        info="Maximum sentences per scene description."
                    )
                style_suffix_input = gr.Textbox(
                    label="Style Suffix",
                    value=DEFAULT_STYLE_SUFFIX,
                    info="Style keywords appended to scene descriptions for consistent visual style."
                )
        
        with gr.TabItem("Video Settings"):
            with gr.Column():
                with gr.Row():
                    template_dropdown = gr.Dropdown(
                        label="Subtitle Style",
                        choices=template_choices,
                        value=DEFAULT_TEMPLATE,
                        info="Style template for kinetic subtitles."
                    )
                    res_dropdown = gr.Dropdown(
                        label="Resolution",
                        choices=["1024x576", "512x288", "1280x720"],
                        value=DEFAULT_RESOLUTION,
                        info="Higher resolution = better quality but slower generation."
                    )
                with gr.Row():
                    fps_input = gr.Textbox(
                        label="FPS (frames/sec)",
                        value=DEFAULT_FPS_MODE,
                        info="Set a fixed FPS for final video, or 'Auto' to match lyrics timing."
                    )
                    seed_input = gr.Number(
                        label="Seed (0 = random)",
                        value=DEFAULT_SEED,
                        precision=0,
                        info="Set seed for reproducible results, or 0 for random generation."
                    )
                with gr.Row():
                    image_mode_input = gr.Radio(
                        label="Image Generation Mode",
                        choices=IMAGE_MODES,
                        value=DEFAULT_IMAGE_MODE,
                        info="Independent: each scene is unique. Consistent: each scene is influenced by the previous for style consistency."
                    )
                    strength_slider = gr.Slider(
                        label="Style Consistency Strength",
                        minimum=0.1,
                        maximum=0.9,
                        step=0.05,
                        value=0.5,
                        visible=False,
                        info="Lower values preserve more of the reference image style (only used in Consistent mode)."
                    )
                crossfade_slider = gr.Slider(
                    label="Crossfade Duration (seconds)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=DEFAULT_CROSSFADE,
                    info="Duration of crossfade between scenes. 0 for hard cuts."
                )
    
    # Make strength slider visible only when Consistent mode is selected
    def update_strength_visibility(mode):
        return gr.update(visible=(mode == "Consistent (Img2Img)"))
    
    image_mode_input.change(update_strength_visibility, inputs=image_mode_input, outputs=strength_slider)

    # Preview section
    preview_btn = gr.Button("Preview First Scene")
    with gr.Row(visible=False) as preview_row:
        preview_img = gr.Image(label="Preview Image", type="pil")
        preview_prompt = gr.Textbox(label="Scene Prompt", lines=3)

    # Main generation button
    generate_btn = gr.Button("Generate Music Video", variant="primary")
    
    # Progress indicator
    progress_bar = gr.Progress()
    
    # Results section
    output_video = gr.Video(label="Generated Video", format="mp4")
    download_file = gr.File(label="Download Video", file_count="single")
    
    # Scene info accordion
    prompt_display = gr.Accordion("Generated Scene Prompts", open=False)
    prompt_text = gr.Markdown("", elem_id="prompt_markdown")
    
    # Image gallery for inspection
    image_gallery = gr.Gallery(
        label="Generated Scene Images",
        columns=3,
        rows=2,
        height="auto",
        object_fit="contain"
    )
    
    # Preview function
    def on_preview(
        audio, whisper_model, llm_model, image_model, 
        prompt_template, max_words, max_sentences, style_suffix, resolution
    ):
        if not audio:
            return gr.update(visible=False), None, "Please upload audio first"
        
        # Quick transcription and segmentation of first few seconds
        try:
            # Extract first 5 seconds of audio for quick preview
            import subprocess
            import tempfile
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()
            subprocess.run([
                "ffmpeg", "-y", "-i", audio, "-ss", "0", "-t", "5", "-acodec", "pcm_s16le", temp_audio.name
            ], check=True, capture_output=True)
            
            # Transcribe and get first segment
            result = transcribe_audio(temp_audio.name, "tiny")  # Use tiny model for speed
            segments = segment_lyrics(result)
            os.unlink(temp_audio.name)
            
            if not segments:
                return gr.update(visible=False), None, "No speech detected in first 5 seconds"
            
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
            
            return gr.update(visible=True), image, scene_prompt
            
        except Exception as e:
            print(f"Preview error: {e}")
            return gr.update(visible=False), None, f"Preview failed: {str(e)}"

    # Bind button click to processing function
    def on_generate(
        audio, whisper_model, llm_model, image_model, video_model,
        template_name, resolution, fps, seed, prompt_template,
        max_words, max_sentences, style_suffix, image_mode, strength,
        crossfade_duration, progress=gr.Progress()
    ):
        if not audio:
            return None, None, gr.update(value="", visible=False), []
            
        try:
            # Progress callback function that updates the Gradio progress bar
            def update_progress(percent, desc=""):
                progress(percent / 100, desc)
            
            final_path, work_dir = process_audio(
                audio, whisper_model, llm_model, image_model, video_model,
                template_name, resolution, fps, int(seed), prompt_template,
                max_words, max_sentences, style_suffix, image_mode, strength,
                crossfade_duration, progress=update_progress
            )
            
            # Load prompts from file to display in accordion
            prompts_file = os.path.join(work_dir, "prompts.txt")
            prompts_markdown = ""
            try:
                with open(prompts_file, 'r', encoding='utf-8') as pf:
                    content = pf.read()
                    # Format prompts as Markdown list
                    prompts_lines = content.strip().splitlines()
                    prompts_markdown = "\n".join([f"- {line}" for line in prompts_lines])
            except:
                prompts_markdown = "Scene prompts not available"
                
            # Load generated images for the gallery
            images = []
            try:
                import glob
                image_files = glob.glob(os.path.join(work_dir, "*.png"))
                for img_file in sorted(image_files):
                    if "segment" in os.path.basename(img_file):
                        try:
                            img = Image.open(img_file)
                            images.append(img)
                        except:
                            pass
            except Exception as e:
                print(f"Error loading images for gallery: {e}")
                
            return final_path, final_path, gr.update(value=prompts_markdown, visible=True), images
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, gr.update(value=f"**Error:** {str(e)}", visible=True), []

    preview_btn.click(
        on_preview,
        inputs=[
            audio_input, whisper_model, llm_dropdown, image_dropdown, 
            prompt_template_input, max_words_input, max_sentences_input,
            style_suffix_input, res_dropdown
        ],
        outputs=[preview_row, preview_img, preview_prompt]
    )

    generate_btn.click(
        on_generate,
        inputs=[
            audio_input, whisper_dropdown, llm_dropdown, image_dropdown, video_dropdown,
            template_dropdown, res_dropdown, fps_input, seed_input, prompt_template_input,
            max_words_input, max_sentences_input, style_suffix_input,
            image_mode_input, strength_slider, crossfade_slider
        ],
        outputs=[output_video, download_file, prompt_text, image_gallery]
    )

if __name__ == "__main__":
    # Uncomment for custom hosting options
    # demo.launch(server_name='0.0.0.0', server_port=7860)
    demo.launch()