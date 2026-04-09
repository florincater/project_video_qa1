import cv2
import subprocess
import os
from pathlib import Path

class VideoIngestion:
    def __init__(self):
        pass
    
    def validate_video(self, video_path):
        """Validate if the video file can be processed"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            # Check if video has at least one frame
            ret, frame = cap.read()
            if not ret:
                return False, "Video has no frames"
            
            cap.release()
            return True, "Video is valid"
        except Exception as e:
            return False, f"Error validating video: {str(e)}"
    
    def extract_metadata(self, video_path):
        """Extract video metadata using OpenCV"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
    
    def extract_audio(self, video_path):
        """Extract audio from video using ffmpeg"""
        audio_path = video_path.replace('.mp4', '_audio.wav')
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-acodec', 'pcm_s16le',
            '-ac', '1',  # mono
            '-ar', '16000',  # 16kHz sample rate
            '-y',  # overwrite output file
            audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return audio_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to extract audio: {e.stderr}")