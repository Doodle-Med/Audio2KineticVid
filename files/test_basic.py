#!/usr/bin/env python3
"""
Basic test script for Audio2KineticVid components without requiring model downloads.
Tests the core logic and imports.
"""

def test_segment_logic():
    """Test the segment logic with mock transcription data"""
    print("Testing segment logic...")
    
    # Create mock transcription result similar to Whisper output
    mock_transcription = {
        "text": "Hello world this is a test song with multiple segments and some pauses here and there",
        "segments": [
            {
                "text": " Hello world this is a test",
                "start": 0.0,
                "end": 2.5,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                    {"word": "this", "start": 1.0, "end": 1.3},
                    {"word": "is", "start": 1.3, "end": 1.5},
                    {"word": "a", "start": 1.5, "end": 1.7},
                    {"word": "test", "start": 1.7, "end": 2.5}
                ]
            },
            {
                "text": " song with multiple segments",
                "start": 2.8,
                "end": 5.2,
                "words": [
                    {"word": "song", "start": 2.8, "end": 3.2},
                    {"word": "with", "start": 3.2, "end": 3.5},
                    {"word": "multiple", "start": 3.5, "end": 4.2},
                    {"word": "segments", "start": 4.2, "end": 5.2}
                ]
            },
            {
                "text": " and some pauses here and there",
                "start": 5.5,
                "end": 8.0,
                "words": [
                    {"word": "and", "start": 5.5, "end": 5.7},
                    {"word": "some", "start": 5.7, "end": 6.0},
                    {"word": "pauses", "start": 6.0, "end": 6.5},
                    {"word": "here", "start": 6.5, "end": 6.8},
                    {"word": "and", "start": 6.8, "end": 7.0},
                    {"word": "there", "start": 7.0, "end": 8.0}
                ]
            }
        ]
    }
    
    try:
        from utils.segment import segment_lyrics, get_segment_info
        
        # Test segmentation
        segments = segment_lyrics(mock_transcription)
        print(f"✅ Segmented into {len(segments)} segments")
        
        # Test segment info
        info = get_segment_info(segments)
        print(f"✅ Segment info: {info['total_segments']} segments, {info['total_duration']:.1f}s total")
        
        # Print segments for inspection
        for i, seg in enumerate(segments):
            duration = seg['end'] - seg['start']
            print(f"   Segment {i+1}: '{seg['text'][:30]}...' ({duration:.1f}s)")
            
        return True
        
    except Exception as e:
        print(f"❌ Segment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test our new segment module
        from utils.segment import segment_lyrics, get_segment_info
        print("✅ segment.py imports successfully")
        
        # Test other modules (without actually calling model-dependent functions)
        import utils.transcribe
        print("✅ transcribe.py imports successfully")
        
        import utils.prompt_gen
        print("✅ prompt_gen.py imports successfully")
        
        import utils.video_gen
        print("✅ video_gen.py imports successfully")
        
        import utils.glue
        print("✅ glue.py imports successfully")
        
        # Test function lists (these shouldn't require models to be loaded)
        whisper_models = utils.transcribe.list_available_whisper_models()
        print(f"✅ {len(whisper_models)} Whisper models available")
        
        llm_models = utils.prompt_gen.list_available_llm_models()
        print(f"✅ {len(llm_models)} LLM models available")
        
        image_models = utils.video_gen.list_available_image_models()
        print(f"✅ {len(image_models)} Image models available")
        
        video_models = utils.video_gen.list_available_video_models()
        print(f"✅ {len(video_models)} Video models available")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_structure():
    """Test that the main app can be imported and has expected structure"""
    print("Testing app structure...")
    
    try:
        # Try to import the main app module
        import app
        print("✅ app.py imports successfully")
        
        # Check if Gradio interface exists
        if hasattr(app, 'demo'):
            print("✅ Gradio demo interface found")
        else:
            print("❌ Gradio demo interface not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ App structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_templates():
    """Test that templates are properly structured"""
    print("Testing template structure...")
    
    import os
    import json
    
    try:
        # Check minimalist template
        minimalist_path = "templates/minimalist"
        if os.path.exists(minimalist_path):
            print("✅ Minimalist template folder exists")
            
            # Check template files
            template_json = os.path.join(minimalist_path, "pycaps.template.json")
            styles_css = os.path.join(minimalist_path, "styles.css")
            
            if os.path.exists(template_json):
                print("✅ Template JSON exists")
                # Validate JSON structure
                with open(template_json) as f:
                    template_data = json.load(f)
                    if 'template_name' in template_data:
                        print("✅ Template JSON has valid structure")
                    else:
                        print("❌ Template JSON missing required fields")
                        return False
            else:
                print("❌ Template JSON missing")
                return False
                
            if os.path.exists(styles_css):
                print("✅ Template CSS exists")
            else:
                print("❌ Template CSS missing")
                return False
        else:
            print("❌ Minimalist template folder missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running Audio2KineticVid basic tests...\n")
    
    tests = [
        test_imports,
        test_segment_logic, 
        test_templates,
        test_app_structure,
    ]
    
    results = []
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        success = test()
        results.append(success)
        print("")
    
    passed = sum(results)
    total = len(results)
    
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application structure is complete.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)