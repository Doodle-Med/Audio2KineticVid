"""
Audio segment processing for creating meaningful lyric segments for video generation.
This module takes Whisper transcription results and intelligently segments them
at natural pause points for synchronized video scene changes.
"""

import re
from typing import List, Dict, Any


def segment_lyrics(transcription_result: Dict[str, Any], min_segment_duration: float = 2.0, max_segment_duration: float = 8.0) -> List[Dict[str, Any]]:
    """
    Segment the transcription into meaningful chunks for video generation.
    
    This function takes the raw Whisper transcription and creates logical segments
    by identifying natural pause points in the audio. Each segment represents
    a coherent lyrical phrase that will correspond to one video scene.
    
    Args:
        transcription_result: Dictionary from Whisper transcription containing 'segments'
        min_segment_duration: Minimum duration for a segment in seconds
        max_segment_duration: Maximum duration for a segment in seconds
        
    Returns:
        List of segment dictionaries with keys:
        - 'text': The lyrical text for this segment
        - 'start': Start time in seconds
        - 'end': End time in seconds  
        - 'words': List of word-level timestamps (if available)
    """
    if not transcription_result or 'segments' not in transcription_result:
        return []
    
    raw_segments = transcription_result['segments']
    if not raw_segments:
        return []
    
    # First, merge very short segments and split very long ones
    processed_segments = []
    
    for segment in raw_segments:
        duration = segment.get('end', 0) - segment.get('start', 0)
        text = segment.get('text', '').strip()
        
        if duration < min_segment_duration:
            # Try to merge with previous segment if it exists and won't exceed max duration
            if (processed_segments and 
                (processed_segments[-1]['end'] - processed_segments[-1]['start'] + duration) <= max_segment_duration):
                # Merge with previous segment
                processed_segments[-1]['text'] += ' ' + text
                processed_segments[-1]['end'] = segment.get('end', processed_segments[-1]['end'])
                if 'words' in segment and 'words' in processed_segments[-1]:
                    processed_segments[-1]['words'].extend(segment['words'])
            else:
                # Add as new segment even if short
                processed_segments.append({
                    'text': text,
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'words': segment.get('words', [])
                })
        elif duration > max_segment_duration:
            # Split long segments at natural break points
            split_segments = _split_long_segment(segment, max_segment_duration)
            processed_segments.extend(split_segments)
        else:
            # Duration is just right
            processed_segments.append({
                'text': text,
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'words': segment.get('words', [])
            })
    
    # Second pass: apply intelligent segmentation based on content
    final_segments = _apply_intelligent_segmentation(processed_segments, max_segment_duration)
    
    # Ensure no empty segments
    final_segments = [seg for seg in final_segments if seg['text'].strip()]
    
    return final_segments


def _split_long_segment(segment: Dict[str, Any], max_duration: float) -> List[Dict[str, Any]]:
    """
    Split a long segment into smaller ones at natural break points.
    """
    text = segment.get('text', '').strip()
    words = segment.get('words', [])
    start_time = segment.get('start', 0)
    end_time = segment.get('end', 0)
    duration = end_time - start_time
    
    if not words or duration <= max_duration:
        return [segment]
    
    # Try to split at punctuation marks or word boundaries
    split_points = []
    
    # Find punctuation-based split points
    for i, word in enumerate(words):
        word_text = word.get('word', '').strip()
        if re.search(r'[.!?;,:]', word_text):
            split_points.append(i)
    
    # If no punctuation, split at word boundaries roughly evenly
    if not split_points:
        target_splits = int(duration / max_duration)
        words_per_split = len(words) // (target_splits + 1)
        split_points = [i * words_per_split for i in range(1, target_splits + 1) if i * words_per_split < len(words)]
    
    if not split_points:
        return [segment]
    
    # Create segments from split points
    segments = []
    last_idx = 0
    
    for split_idx in split_points:
        if split_idx >= len(words):
            continue
            
        segment_words = words[last_idx:split_idx + 1]
        if segment_words:
            segments.append({
                'text': ' '.join([w.get('word', '') for w in segment_words]).strip(),
                'start': segment_words[0].get('start', start_time),
                'end': segment_words[-1].get('end', end_time),
                'words': segment_words
            })
        last_idx = split_idx + 1
    
    # Add remaining words as final segment
    if last_idx < len(words):
        segment_words = words[last_idx:]
        segments.append({
            'text': ' '.join([w.get('word', '') for w in segment_words]).strip(),
            'start': segment_words[0].get('start', start_time),
            'end': segment_words[-1].get('end', end_time),
            'words': segment_words
        })
    
    return segments


def _apply_intelligent_segmentation(segments: List[Dict[str, Any]], max_duration: float) -> List[Dict[str, Any]]:
    """
    Apply intelligent segmentation rules based on lyrical content and timing.
    """
    if not segments:
        return []
    
    final_segments = []
    current_segment = None
    
    for segment in segments:
        text = segment['text'].strip()
        
        # Skip empty segments
        if not text:
            continue
        
        # If no current segment, start a new one
        if current_segment is None:
            current_segment = segment.copy()
            continue
        
        # Check if we should merge with current segment
        should_merge = _should_merge_segments(current_segment, segment, max_duration)
        
        if should_merge:
            # Merge segments
            current_segment['text'] += ' ' + segment['text']
            current_segment['end'] = segment['end']
            if 'words' in segment and 'words' in current_segment:
                current_segment['words'].extend(segment['words'])
        else:
            # Finalize current segment and start new one
            final_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    if current_segment is not None:
        final_segments.append(current_segment)
    
    return final_segments


def _should_merge_segments(current: Dict[str, Any], next_seg: Dict[str, Any], max_duration: float) -> bool:
    """
    Determine if two segments should be merged based on content and timing.
    """
    # Check duration constraint
    merged_duration = next_seg['end'] - current['start']
    if merged_duration > max_duration:
        return False
    
    current_text = current['text'].strip()
    next_text = next_seg['text'].strip()
    
    # Don't merge if current segment ends with strong punctuation
    if re.search(r'[.!?]$', current_text):
        return False
    
    # Merge if current segment is very short (likely incomplete phrase)
    if len(current_text.split()) < 3:
        return True
    
    # Merge if next segment starts with a lowercase word (continuation)
    if next_text and next_text[0].islower():
        return True
    
    # Merge if there's a short gap between segments (< 0.5 seconds)
    gap = next_seg['start'] - current['end']
    if gap < 0.5:
        return True
    
    # Don't merge by default
    return False


def get_segment_info(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get summary information about the segments.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        Dictionary with segment statistics
    """
    if not segments:
        return {
            'total_segments': 0,
            'total_duration': 0,
            'average_duration': 0,
            'shortest_duration': 0,
            'longest_duration': 0
        }
    
    durations = [seg['end'] - seg['start'] for seg in segments]
    total_duration = segments[-1]['end'] - segments[0]['start'] if segments else 0
    
    return {
        'total_segments': len(segments),
        'total_duration': total_duration,
        'average_duration': sum(durations) / len(durations),
        'shortest_duration': min(durations),
        'longest_duration': max(durations),
        'segments_preview': [{'text': seg['text'][:50] + '...', 'duration': seg['end'] - seg['start']} for seg in segments[:5]]
    }