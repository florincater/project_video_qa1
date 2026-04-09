from typing import Dict, List
import os
import streamlit as st
from vector_store import VideoVectorStore

class VideoRAG:
    def __init__(self, vector_store: VideoVectorStore, model="gpt-3.5-turbo"):
        """Initialize RAG system with vector store and LLM"""
        self.vector_store = vector_store
        self.model = model
        
        # Try to get API key from Streamlit secrets or environment variable
        self.api_key = None
        try:
            # For Streamlit Cloud
            self.api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            # For local development
            self.api_key = os.getenv("OPENAI_API_KEY")
    
    def answer(self, question: str, k: int = 3) -> Dict:
        """Answer question using RAG"""
        # Search for relevant context
        relevant_chunks = self.vector_store.search(question, k=k)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information in the video to answer your question.",
                'sources': []
            }
        
        # Prepare context from relevant chunks
        context = self._prepare_context(relevant_chunks)
        
        # Check if we have an API key
        if not self.api_key:
            # Return restricted mode response
            return {
                'answer': self._get_restricted_response(question, context),
                'sources': relevant_chunks,
                'restricted_mode': True
            }
        
        # Generate answer using LLM (full mode)
        answer = self._generate_answer(question, context)
        
        return {
            'answer': answer,
            'sources': relevant_chunks,
            'restricted_mode': False
        }
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from chunks for LLM"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Segment {i} - {chunk['start']:.1f}s to {chunk['end']:.1f}s]\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def _get_restricted_response(self, question: str, context: str) -> str:
        """Generate response for restricted mode (no API key)"""
        # Extract the most relevant text snippet
        lines = context.split('\n')
        first_chunk = lines[0] if lines else ""
        
        return f"""🔒 **Restricted Mode Active**

To get AI-powered detailed answers, please set up your OpenAI API key.

**Here's what was found in the video:**
{context}

**Based on your question:** "{question}"

The relevant segments above contain information that would answer your question. 
To enable full AI analysis, add your OpenAI API key to Streamlit Cloud secrets.

*Note: The app is working correctly, but AI features require an API key.*"""

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI API (requires API key)"""
        import openai
        
        # Set the API key
        openai.api_key = self.api_key
        
        prompt = f"""Based on the following video transcript segments, please answer the question concisely and accurately.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            # Using the new OpenAI v1.0+ API format
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions based on video transcripts. Provide accurate, concise answers based only on the context provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback if API call fails
            return f"Error generating answer: {str(e)}\n\nHere's the relevant context from the video:\n\n{context}"