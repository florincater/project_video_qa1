# app.py
import streamlit as st
import tempfile
from ingestion import VideoIngestion
from transcription import VideoTranscriber
from vector_store import VideoVectorStore
from rag import VideoRAG

st.title("🎥 Video Q&A System")
st.markdown("Posez des questions sur le contenu de vos vidéos !")

# Initialisation des composants
@st.cache_resource
def init_components():
    ingestion = VideoIngestion()
    transcriber = VideoTranscriber(model_size="base")
    vector_store = VideoVectorStore()
    rag = VideoRAG(vector_store)
    return ingestion, transcriber, vector_store, rag

ingestion, transcriber, vector_store, rag = init_components()

# Upload vidéo
uploaded_file = st.file_uploader(
    "Choisissez une vidéo", 
    type=['mp4', 'mkv', 'mov', 'webm']
)

if uploaded_file:
    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Affichage de la vidéo
    st.video(video_path)
    
    # Traitement
    with st.spinner("Analyse de la vidéo en cours..."):
        # Validation
        is_valid, msg = ingestion.validate_video(video_path)
        if not is_valid:
            st.error(msg)
        else:
            # Métadonnées
            metadata = ingestion.extract_metadata(video_path)
            st.info(f"Durée: {metadata['duration']:.1f}s, FPS: {metadata['fps']}")
            
            # Extraction audio
            audio_path = ingestion.extract_audio(video_path)
            
            # Transcription
            transcription = transcriber.transcribe(audio_path)
            chunks = transcriber.chunk_transcription(transcription)
            
            # Vectorisation
            vector_store.add_chunks("video_1", chunks)
            
            st.success(f"✅ Vidéo traitée ! {len(chunks)} chunks créés.")
    
    # Zone de questions
    question = st.text_input("Posez votre question sur la vidéo :")
    
    if question:
        with st.spinner("Recherche de la réponse..."):
            response = rag.answer(question)
            
            # 🔽 MODE INDICATOR GOES HERE - Right after getting response 🔽
            if response.get('restricted_mode', False):
                st.info("ℹ️ **Mode restreint actif** - Fonctionnalités IA complètes disponibles avec clé API OpenAI")
            # 🔼 MODE INDICATOR ENDS HERE 🔼
            
            st.markdown("### Réponse")
            st.write(response['answer'])
            
            st.markdown("### Sources")
            for source in response['sources']:
                st.caption(f"[{source['start']:.1f}s - {source['end']:.1f}s] {source['text'][:100]}...")