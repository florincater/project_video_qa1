import whisper
from typing import List, Dict

class VideoTranscriber:
    def __init__(self, model_size="base"):
        """Initialize Whisper model"""
        self.model = whisper.load_model(model_size)
    
    def transcribe(self, audio_path):
        """Transcribe audio using Whisper"""
        result = self.model.transcribe(audio_path)
        return result
    
    def chunk_transcription(self, transcription, chunk_duration=30):
        """Split transcription into manageable chunks"""
        segments = transcription.get('segments', [])
        chunks = []
        
        current_chunk = {
            'text': '',
            'start': 0,
            'end': 0,
            'segments': []
        }
        
        for segment in segments:
            # If adding this segment would exceed chunk duration, save current chunk
            if (segment['end'] - current_chunk['start']) > chunk_duration and current_chunk['text']:
                chunks.append(current_chunk)
                current_chunk = {
                    'text': '',
                    'start': segment['start'],
                    'end': 0,
                    'segments': []
                }
            
            # Add segment to current chunk
            current_chunk['text'] += segment['text'] + ' '
            current_chunk['end'] = segment['end']
            current_chunk['segments'].append(segment)
        
        # Add last chunk
        if current_chunk['text']:
            chunks.append(current_chunk)
        
        return chunks