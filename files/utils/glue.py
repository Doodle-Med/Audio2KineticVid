import os
import subprocess
import json

def stitch_and_caption(
    segment_videos, 
    audio_path, 
    transcription_segments, 
    template_name, 
    work_dir=".",
    crossfade_duration=0.25
):
    """
    Stitch video segments with crossfade transitions, add original audio, and overlay kinetic captions.
    
    Args:
        segment_videos (list): List of file paths for the video segments.
        audio_path (str): Path to the original audio file.
        transcription_segments (list): The list of segment dictionaries from segment.py, including text and word timestamps.
        template_name (str): The name of the PyCaps template to use.
        work_dir (str): The working directory for temporary and final files.
        crossfade_duration (float): Duration of crossfade transitions in seconds (0 for hard cuts).

    Returns:
        str: The path to the final subtitled video.
    """
    if not segment_videos:
        raise RuntimeError("No video segments to stitch.")
    
    stitched_path = os.path.join(work_dir, "stitched.mp4")
    final_path = os.path.join(work_dir, "final_video.mp4")

    # 1. Stitch video segments together with crossfades using ffmpeg
    print("Stitching video segments with crossfades...")
    try:
        # Get accurate durations for each video segment using ffprobe
        durations = [_get_video_duration(seg_file) for seg_file in segment_videos]
        
        cross_dur = crossfade_duration  # Crossfade duration in seconds
        
        # Handle the case where crossfade is disabled (hard cuts)
        if cross_dur <= 0:
            # Use concat demuxer for hard cuts (more reliable for exact segment timing)
            concat_file = os.path.join(work_dir, "concat_list.txt")
            with open(concat_file, "w") as f:
                for seg_file in segment_videos:
                    f.write(f"file '{os.path.abspath(seg_file)}'\n")
            
            # Run ffmpeg with concat demuxer
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-i", audio_path,
                "-c:v", "copy",  # Copy video stream without re-encoding for speed
                "-c:a", "aac",
                "-b:a", "192k",
                "-map", "0:v",
                "-map", "1:a",
                "-shortest",
                stitched_path
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        else:
            # Build the complex filter string for ffmpeg with crossfades
            inputs = []
            filter_complex_parts = []
            stream_labels = []

            # Prepare inputs and initial stream labels
            for i, seg_file in enumerate(segment_videos):
                inputs.extend(["-i", seg_file])
                stream_labels.append(f"[{i}:v]")

            # If only one video, no stitching needed, just prep for subtitling
            if len(segment_videos) == 1:
                final_video_stream = "[0:v]"
                filter_complex_str = f"[0:v]format=yuv420p[video]"
            else:
                # Sequentially chain xfade filters
                last_stream_label = stream_labels[0]
                current_offset = 0.0
                
                for i in range(len(segment_videos) - 1):
                    current_offset += durations[i] - cross_dur
                    next_stream_label = f"v{i+1}"
                    
                    filter_complex_parts.append(
                        f"{last_stream_label}{stream_labels[i+1]}"
                        f"xfade=transition=fade:duration={cross_dur}:offset={current_offset}"
                        f"[{next_stream_label}]"
                    )
                    last_stream_label = f"[{next_stream_label}]"
                
                final_video_stream = last_stream_label
                filter_complex_str = ";".join(filter_complex_parts)
                filter_complex_str += f";{final_video_stream}format=yuv420p[video]"

            # Construct the full ffmpeg command
            cmd = ["ffmpeg", "-y"]
            cmd.extend(inputs)
            cmd.extend(["-i", audio_path]) # Add original audio as the last input
            cmd.extend([
                "-filter_complex", filter_complex_str,
                "-map", "[video]",                             # Map the final video stream
                "-map", f"{len(segment_videos)}:a",             # Map the audio stream
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",                                   # Finish encoding when the shortest stream ends
                stitched_path
            ])

            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
    except subprocess.CalledProcessError as e:
        print("Error during ffmpeg stitching:")
        print("FFMPEG stdout:", e.stdout)
        print("FFMPEG stderr:", e.stderr)
        raise RuntimeError("FFMPEG stitching failed.") from e

    # 2. Use PyCaps to render captions on the stitched video
    print("Overlaying kinetic subtitles...")
    
    # Save the real transcription data to a JSON file for PyCaps
    transcription_json_path = os.path.join(work_dir, "transcription_for_pycaps.json")
    _save_whisper_json(transcription_segments, transcription_json_path)

    # Run pycaps render command
    try:
        pycaps_cmd = [
            "pycaps", "render",
            "--input", stitched_path,
            "--template", os.path.join("templates", template_name),
            "--whisper-json", transcription_json_path,
            "--output", final_path
        ]
        subprocess.run(pycaps_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("`pycaps` command not found. Make sure pycaps is installed correctly (e.g., `pip install git+https://github.com/francozanardi/pycaps.git`).")
    except subprocess.CalledProcessError as e:
        print("Error during PyCaps subtitle rendering:")
        print("PyCaps stdout:", e.stdout)
        print("PyCaps stderr:", e.stderr)
        raise RuntimeError("PyCaps rendering failed.") from e
        
    return final_path


def _get_video_duration(file_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        return float(output)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not get duration for {file_path}. Error: {e}. Falling back to 0.0.")
        return 0.0


def _save_whisper_json(transcription_segments, json_path):
    """
    Saves the transcription segments into a Whisper-formatted JSON file for PyCaps.
    
    Args:
        transcription_segments (list): A list of segment dictionaries, each containing
                                       'start', 'end', 'text', and 'words' keys.
        json_path (str): The file path to save the JSON data.
    """
    print(f"Saving transcription to {json_path} for subtitling...")
    # The structure pycaps expects is a dictionary with a "segments" key,
    # which contains the list of segment dictionaries.
    output_data = {
        "text": " ".join([seg.get('text', '') for seg in transcription_segments]),
        "segments": transcription_segments,
        "language": "en"
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write transcription JSON file at {json_path}") from e