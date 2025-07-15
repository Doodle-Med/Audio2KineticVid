#!/usr/bin/env bash
# Smoke test: generate a video for a short demo audio clip (30s)
# Ensure ffmpeg is installed and the environment has the required models downloaded.

# Use a sample audio (30s) - replace with actual file path if needed
DEMO_AUDIO=${1:-demo.mp3}

if [ ! -f "$DEMO_AUDIO" ]; then
  echo "Demo audio file not found: $DEMO_AUDIO"
  exit 1
fi

# Run transcription
echo "Transcribing $DEMO_AUDIO..."
python -c "from utils.transcribe import transcribe_audio; import json, sys; result = transcribe_audio('$DEMO_AUDIO', 'base'); print(json.dumps(result, indent=2))" > transcription.json

# Run segmentation
echo "Segmenting lyrics..."
python -c "import json; from utils.segment import segment_lyrics; data=json.load(open('transcription.json')); segments=segment_lyrics(data); json.dump(segments, open('segments.json','w'), indent=2)"

# Generate scene prompts
echo "Generating scene prompts..."
python -c "import json; from utils.prompt_gen import generate_scene_prompts; segments=json.load(open('segments.json')); prompts=generate_scene_prompts(segments); json.dump(prompts, open('prompts.json','w'), indent=2)"

# Generate video segments
echo "Generating video segments..."
python -c "import json; from utils import video_gen; segments=json.load(open('segments.json')); prompts=json.load(open('prompts.json')); files=video_gen.create_video_segments(segments, prompts, width=512, height=288, dynamic_fps=True, seed=42, work_dir='tmp/smoke_test'); print(json.dumps(files, indent=2))" > segment_files.json

# Stitch and add captions - UPDATED with segments parameter
echo "Stitching segments and adding subtitles..."
python -c "import json; from utils.glue import stitch_and_caption; files=json.load(open('segment_files.json')); segments=json.load(open('segments.json')); out=stitch_and_caption(files, '$DEMO_AUDIO', segments, 'minimalist', work_dir='tmp/smoke_test'); print('Final video saved to:', out)"

# The final video will be tmp/smoke_test/final.mp4