import os
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from PIL import Image
import numpy as np
import time

# Global pipelines cache
_model_cache = {}

def list_available_image_models():
    """Return list of available image generation models"""
    return [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1"
    ]

def list_available_video_models():
    """Return list of available video generation models"""
    return [
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "stabilityai/stable-video-diffusion-img2vid"
    ]

def _get_model_key(model_name, is_img2img=False):
    """Generate a unique key for the model cache"""
    return f"{model_name}_{'img2img' if is_img2img else 'txt2img'}"

def _load_image_pipeline(model_name, is_img2img=False):
    """Load image generation pipeline with caching"""
    model_key = _get_model_key(model_name, is_img2img)
    
    if model_key not in _model_cache:
        print(f"Loading image model: {model_name} ({is_img2img})")
        
        if "xl" in model_name.lower():
            # SDXL model
            if is_img2img:
                pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )
            else:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True
                )
        else:
            # SD 1.5/2.x model
            if is_img2img:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                )
                
        pipeline.enable_model_cpu_offload()
        pipeline.safety_checker = None  # disable safety checker for performance
        _model_cache[model_key] = pipeline
        
    return _model_cache[model_key]

def _load_video_pipeline(model_name):
    """Load video generation pipeline with caching"""
    if model_name not in _model_cache:
        print(f"Loading video model: {model_name}")
        
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipeline.enable_model_cpu_offload()
        
        # Enable forward chunking for lower VRAM use
        pipeline.unet.enable_forward_chunking(chunk_size=1)
        
        _model_cache[model_name] = pipeline
        
    return _model_cache[model_name]

def preview_image_generation(prompt, image_model="stabilityai/stable-diffusion-xl-base-1.0", width=1024, height=576, seed=None):
    """
    Generate a preview image from a prompt
    
    Args:
        prompt: Text prompt for image generation
        image_model: Model to use
        width/height: Image dimensions
        seed: Random seed (None for random)
    
    Returns:
        PIL Image object
    """
    pipeline = _load_image_pipeline(image_model)
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    
    with torch.autocast("cuda"):
        image = pipeline(
            prompt,
            width=width,
            height=height,
            generator=generator,
            num_inference_steps=30
        ).images[0]
    
    return image

def create_video_segments(
    segments,
    scene_prompts,
    image_model="stabilityai/stable-diffusion-xl-base-1.0",
    video_model="stabilityai/stable-video-diffusion-img2vid-xt",
    width=1024,
    height=576,
    dynamic_fps=True,
    base_fps=None,
    seed=None,
    work_dir=".",
    image_mode="Independent",
    strength=0.5,
    progress_callback=None
):
    """
    Generate an image and a short video clip for each segment.
    
    Args:
        segments: List of segment dictionaries with timing info
        scene_prompts: List of text prompts for each segment
        image_model: Model to use for image generation
        video_model: Model to use for video generation
        width/height: Video dimensions
        dynamic_fps: If True, adjust FPS to match segment duration
        base_fps: Base FPS when dynamic_fps is False
        seed: Random seed (None or 0 for random)
        work_dir: Directory to save intermediate files
        image_mode: "Independent" or "Consistent (Img2Img)" for style continuity
        strength: Strength parameter for img2img (0-1, lower preserves more reference)
        progress_callback: Function to call with progress updates
        
    Returns:
        List of file paths to the segment video clips
    """
    # Initialize image and video pipelines
    txt2img_pipe = _load_image_pipeline(image_model)
    video_pipe = _load_video_pipeline(video_model)
    
    # Set manual seed if provided
    generator = None
    if seed is not None and int(seed) != 0:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
    
    segment_files = []
    reference_image = None
    
    for idx, (seg, prompt) in enumerate(zip(segments, scene_prompts)):
        if progress_callback:
            progress_percent = (idx / len(segments)) * 100
            progress_callback(progress_percent, f"Generating scene {idx+1}/{len(segments)}")
        
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_dur = max(seg_end - seg_start, 0.001)
        
        # Determine FPS for this segment
        if dynamic_fps:
            # Use 25 frames spanning the segment duration
            fps = 25.0 / seg_dur
            # Cap FPS to 30 to avoid too high frame rate for very short segments
            if fps > 30.0: 
                fps = 30.0
        else:
            fps = base_fps or 10.0  # use given fixed fps, default 10 if not set
        
        # 1. Generate initial frame image with Stable Diffusion
        img_filename = os.path.join(work_dir, f"segment{idx:02d}_img.png")
        
        with torch.autocast("cuda"):
            if image_mode == "Consistent (Img2Img)" and reference_image is not None:
                # Use img2img with reference image for style consistency
                img2img_pipe = _load_image_pipeline(image_model, is_img2img=True)
                image = img2img_pipe(
                    prompt=prompt,
                    image=reference_image,
                    strength=strength,
                    generator=generator,
                    num_inference_steps=30
                ).images[0]
            else:
                # Regular text-to-image generation
                image = txt2img_pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    generator=generator,
                    num_inference_steps=30
                ).images[0]
        
        # Save the image for inspection
        image.save(img_filename)
        
        # Update reference image for next segment if using consistent mode
        if image_mode == "Consistent (Img2Img)":
            reference_image = image
        
        # 2. Generate video frames from the image using stable video diffusion
        with torch.autocast("cuda"):
            video_frames = video_pipe(
                image, 
                num_frames=25,
                fps=fps,
                decode_chunk_size=1,
                generator=generator
            ).frames[0]
        
        # Save video frames to a file (mp4)
        seg_filename = os.path.join(work_dir, f"segment_{idx:03d}.mp4")
        from diffusers.utils import export_to_video
        export_to_video(video_frames, seg_filename, fps=fps)
        segment_files.append(seg_filename)
        
        # Free memory from frames
        del video_frames
        torch.cuda.empty_cache()
        
    # Return list of video segment files
    return segment_files