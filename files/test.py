#!/usr/bin/env python3
"""
Simple test script for Audio2KineticVid components.
This tests each pipeline component individually.
"""

import os
import sys
from PIL import Image

def run_tests():
    print("Testing Audio2KineticVid components...")
    
    # Test for demo audio file
    if not os.path.exists("demo.mp3"):
        print("❌ No demo.mp3 found. Please add a short audio file for testing.")
        print("   Continuing with partial tests...")
    else:
        print("✅ Demo audio file found")
    
    # Test GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ No GPU available! This app requires a CUDA-capable GPU.")
        return False
    
    # Test imports
    try:
        print("Testing imports...")
        import gradio
        import whisper
        import transformers
        import diffusers
        print("✅ All required libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've installed all dependencies: pip install -r requirements.txt")
        return False
    
    # Test module imports
    try:
        print("Testing module imports...")
        from utils.transcribe import list_available_whisper_models
        from utils.prompt_gen import list_available_llm_models
        from utils.video_gen import list_available_image_models
        
        print(f"✅ Available Whisper models: {list_available_whisper_models()[:3]}...")
        print(f"✅ Available LLM models: {list_available_llm_models()[:2]}...")
        print(f"✅ Available Image models: {list_available_image_models()[:2]}...")
    except Exception as e:
        print(f"❌ Module import error: {e}")
        return False
    
    # Test text-to-image (lightweight test)
    try:
        print("Testing image generation (minimal)...")
        from utils.video_gen import preview_image_generation
        
        # Use a very small model for quick testing
        test_image = preview_image_generation(
            "A blue sky with clouds",
            image_model="runwayml/stable-diffusion-v1-5",
            width=256,
            height=256
        )
        
        test_image.save("test_image.png")
        print(f"✅ Generated test image: test_image.png")
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTests completed!")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)