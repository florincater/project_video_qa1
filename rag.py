from typing import Dict, List
import openai
from vector_store import VideoVectorStore

class VideoRAG:
    def __init__(self, vector_store: VideoVectorStore, model="gpt-3.5-turbo"):
        """Initialize RAG system with vector store and LLM"""
        self.vector_store = vector_store
        self.model = model
        
        # Note: Set your OpenAI API key in environment variables
        # openai.api_key = os.getenv("OPENAI_API_KEY")
    
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
        
        # Generate answer using LLM
        answer = self._generate_answer(question, context)
        
        return {
            'answer': answer,
            'sources': relevant_chunks
        }
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from chunks for LLM"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Segment {i} - {chunk['start']:.1f}s to {chunk['end']:.1f}s]\n{chunk['text']}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM (simplified version without API call)"""
        # For now, return a simple response
        # Replace this with actual LLM API call when you have the API key
        
        prompt = f"""Based on the following video transcript segments, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Simple fallback response (replace with actual LLM call)
        # For production, uncomment the OpenAI code:
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant answering questions based on video transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
        """
        
        # Temporary response generation
        return f"Based on the video transcript, I found the following relevant information:\n\n{context}\n\nTo get detailed answers, please set up your OpenAI API key in the RAG class."