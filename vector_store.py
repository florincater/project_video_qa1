from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class VideoVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize vector store with sentence transformer and FAISS"""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = {}
        
    def add_chunks(self, video_id: str, chunks: List[Dict]):
        """Add transcription chunks to vector store"""
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Initialize FAISS index if needed
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                'video_id': video_id,
                'text': chunk['text'],
                'start': chunk['start'],
                'end': chunk['end'],
                'embedding': embeddings[i]
            })
            
            if video_id not in self.metadata:
                self.metadata[video_id] = []
            self.metadata[video_id].append({
                'index': len(self.chunks) - 1,
                'start': chunk['start'],
                'end': chunk['end'],
                'text': chunk['text']
            })
    
    def search(self, query: str, k: int = 5):
        """Search for relevant chunks"""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return relevant chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'score': float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                    'text': self.chunks[idx]['text'],
                    'start': self.chunks[idx]['start'],
                    'end': self.chunks[idx]['end'],
                    'video_id': self.chunks[idx]['video_id']
                })
        
        return results
    
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.chunks = []
        self.metadata = {}