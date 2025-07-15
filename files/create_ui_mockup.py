"""
UI Mockup Generator for Audio2KineticVid
Creates a visual representation of the improved user interface
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_ui_mockup():
    """Create a mockup of the improved Audio2KineticVid interface"""
    
    # Create a large canvas
    width, height = 1200, 1600
    img = Image.new('RGB', (width, height), color='#f8f9fa')
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        normal_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        normal_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    y = 20
    
    # Header
    draw.rectangle([0, 0, width, 80], fill='#2c3e50')
    draw.text((20, 25), "🎵 Audio → Kinetic-Subtitle Music Video", fill='white', font=title_font)
    draw.text((20, 55), "Transform your audio tracks into dynamic music videos with AI", fill='#ecf0f1', font=normal_font)
    
    y = 100
    
    # Features section
    draw.rectangle([20, y, width-20, y+120], outline='#e9ecef', width=2, fill='#ffffff')
    draw.text((30, y+10), "✨ Features", fill='#2c3e50', font=header_font)
    features = [
        "🎤 Whisper Transcription - Accurate speech-to-text",
        "🧠 AI Scene Generation - LLM-powered visual descriptions", 
        "🎨 Image & Video AI - Stable Diffusion + Video Diffusion",
        "🎬 Kinetic Subtitles - Animated text synchronized with audio"
    ]
    for i, feature in enumerate(features):
        draw.text((30, y+35+i*20), feature, fill='#495057', font=normal_font)
    
    y += 140
    
    # Upload section
    draw.rectangle([20, y, width-20, y+80], outline='#007bff', width=2, fill='#e7f3ff')
    draw.text((30, y+10), "🎵 Upload Audio Track", fill='#007bff', font=header_font)
    draw.rectangle([40, y+35, width-40, y+65], outline='#ced4da', width=1, fill='#f8f9fa')
    draw.text((50, y+45), "📁 Choose file... (MP3, WAV, M4A supported)", fill='#6c757d', font=normal_font)
    
    y += 100
    
    # Quality preset section
    draw.rectangle([20, y, width-20, y+100], outline='#28a745', width=2, fill='#e8f5e8')
    draw.text((30, y+10), "⚡ Quality Preset", fill='#28a745', font=header_font)
    presets = ["● Fast (512x288)", "○ Balanced (1024x576)", "○ High Quality (1280x720)"]
    for i, preset in enumerate(presets):
        color = '#28a745' if '●' in preset else '#6c757d'
        draw.text((50, y+35+i*20), preset, fill=color, font=normal_font)
    
    y += 120
    
    # Tabs section
    tabs = ["🤖 AI Models", "✍️ Scene Prompting", "🎬 Video Settings"]
    tab_width = (width - 40) // 3
    for i, tab in enumerate(tabs):
        color = '#007bff' if i == 0 else '#e9ecef'
        text_color = 'white' if i == 0 else '#6c757d'
        draw.rectangle([20 + i*tab_width, y, 20 + (i+1)*tab_width, y+40], fill=color)
        draw.text((30 + i*tab_width, y+15), tab, fill=text_color, font=normal_font)
    
    y += 60
    
    # Models section (active tab)
    draw.rectangle([20, y, width-20, y+200], outline='#007bff', width=2, fill='#ffffff')
    draw.text((30, y+10), "Choose the AI models for each processing step:", fill='#495057', font=normal_font)
    
    # Model dropdowns
    models = [
        ("🎤 Transcription Model", "medium.en (Recommended for English)"),
        ("🧠 Scene Description Model", "Mixtral-8x7B-Instruct (Creative scene generation)"),
        ("🎨 Image Generation Model", "stable-diffusion-xl-base-1.0 (High quality)"),
        ("🎬 Video Animation Model", "stable-video-diffusion-img2vid-xt (Smooth motion)")
    ]
    
    for i, (label, value) in enumerate(models):
        x_offset = 30 + (i % 2) * (width//2 - 40)
        y_offset = y + 40 + (i // 2) * 80
        
        draw.text((x_offset, y_offset), label, fill='#495057', font=normal_font)
        draw.rectangle([x_offset, y_offset+20, x_offset+250, y_offset+45], outline='#ced4da', width=1, fill='#ffffff')
        draw.text((x_offset+5, y_offset+27), value[:35] + "...", fill='#495057', font=small_font)
    
    y += 220
    
    # Action buttons
    button_y = y + 20
    draw.rectangle([40, button_y, 280, button_y+50], fill='#6c757d', outline='#6c757d')
    draw.text((90, button_y+18), "🔍 Preview First Scene", fill='white', font=normal_font)
    
    draw.rectangle([320, button_y, 600, button_y+50], fill='#007bff', outline='#007bff')
    draw.text((370, button_y+18), "🎬 Generate Complete Music Video", fill='white', font=normal_font)
    
    y += 90
    
    # Progress section
    draw.rectangle([20, y, width-20, y+60], outline='#17a2b8', width=2, fill='#e1f7fa')
    draw.text((30, y+10), "📊 Generation Status", fill='#17a2b8', font=header_font)
    draw.text((30, y+35), "✅ Generation complete! (2.3 minutes)", fill='#28a745', font=normal_font)
    
    y += 80
    
    # Results tabs
    result_tabs = ["🎥 Final Video", "🖼️ Generated Images", "📝 Scene Descriptions"]
    tab_width = (width - 40) // 3
    for i, tab in enumerate(result_tabs):
        color = '#28a745' if i == 0 else '#e9ecef'
        text_color = 'white' if i == 0 else '#6c757d'
        draw.rectangle([20 + i*tab_width, y, 20 + (i+1)*tab_width, y+40], fill=color)
        draw.text((30 + i*tab_width, y+15), tab, fill=text_color, font=small_font)
    
    y += 60
    
    # Video result
    draw.rectangle([20, y, width-20, y+150], outline='#28a745', width=2, fill='#ffffff')
    draw.rectangle([30, y+10, width-30, y+120], fill='#000000')
    draw.text((width//2-60, y+60), "🎬 GENERATED VIDEO", fill='white', font=header_font)
    draw.text((30, y+130), "📥 Download: final_video.mp4 (45.2 MB)", fill='#28a745', font=normal_font)
    
    return img

if __name__ == "__main__":
    mockup = create_ui_mockup()
    mockup.save("ui_mockup.png")
    print("✅ UI mockup saved as ui_mockup.png")