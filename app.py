import os
import tempfile
import wave
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
from groq import Groq
from loguru import logger

from simple_math_agent import agent, agent_config

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format=os.getenv("LOG_FORMAT", "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def audio_to_bytes(audio_data, sample_rate=44100):
    """Convert numpy array to audio bytes for API upload."""
    if isinstance(audio_data, tuple):
        sample_rate, audio_array = audio_data
        audio_data = audio_array
    
    if audio_data.dtype != np.int16:
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        audio_data = (audio_data * 32767).astype(np.int16)
    
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    buffer.seek(0)
    return buffer.getvalue()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe audio using Groq Whisper."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        logger.info("🎙️ Received audio for transcription")
        
        # CHANGED: Pass file content as a tuple (filename, bytes) to the API
        transcription_input = (audio_file.filename, audio_file.read())
        
        transcript = groq_client.audio.transcriptions.create(
            file=transcription_input,
            model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )
        
        logger.info(f'👂 Transcribed: "{transcript}"')
        return jsonify({'transcript': transcript})
        
    except Exception as e:
        logger.error(f"❌ Transcription error: {str(e)}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Process chat message through the agent."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f'💬 Processing message: "{message}"')
        
        agent_response = agent.invoke(
            {"messages": [{"role": "user", "content": message}]}, 
            config=agent_config
        )
        response_text = agent_response["messages"][-1].content
        
        logger.info(f'🤖 Agent response: "{response_text}"')
        return jsonify({'response': response_text})
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        return jsonify({'error': f'Chat processing failed: {str(e)}'}), 500


@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Convert text to speech using Groq TTS."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f'🔊 Synthesizing speech for: "{text}"')
        
        tts_response = groq_client.audio.speech.create(
            model=os.getenv("TTS_MODEL", "tts-1"),
            voice=os.getenv("TTS_VOICE", "fable"),
            response_format=os.getenv("TTS_RESPONSE_FORMAT", "mp3"), # CHANGED: Using mp3 for better browser compatibility
            input=text,
        )
        
        audio_bytes = tts_response.read()
        logger.info("🎵 Speech generated successfully")
        
        # CHANGED: Return audio bytes directly for browser playback
        return send_file(
            BytesIO(audio_bytes),
            mimetype='audio/mpeg',
            as_attachment=False
        )
        
    except Exception as e:
        logger.error(f"❌ TTS error: {str(e)}")
        return jsonify({'error': f'Speech synthesis failed: {str(e)}'}), 500


@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    """Complete voice chat pipeline: audio -> transcription -> agent -> TTS."""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        logger.info("🎙️ Processing voice chat request")
        
        # CHANGED: Pass file content as a tuple (filename, bytes) to the API
        transcription_input = (audio_file.filename, audio_file.read())
        
        # Step 1: Transcribe
        transcript = groq_client.audio.transcriptions.create(
            file=transcription_input,
            model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )
        logger.info(f'👂 Transcribed: "{transcript}"')
        
        # Step 2: Process through agent
        agent_response = agent.invoke(
            {"messages": [{"role": "user", "content": transcript}]}, 
            config=agent_config
        )
        response_text = agent_response["messages"][-1].content
        logger.info(f'🤖 Agent response: "{response_text}"')
        
        # Step 3: Generate speech
        tts_response = groq_client.audio.speech.create(
            model=os.getenv("TTS_MODEL", "tts-1"),
            voice=os.getenv("TTS_VOICE", "fable"),
            response_format=os.getenv("TTS_RESPONSE_FORMAT", "mp3"), # CHANGED: Using mp3
            input=response_text,
        )
        
        audio_bytes = tts_response.read()
        
        # CHANGED: Create a response object to send audio and custom headers
        response = Response(BytesIO(audio_bytes), mimetype='audio/mpeg')
        response.headers['X-Transcript'] = transcript
        response.headers['X-Agent-Response'] = response_text
        
        logger.info("✅ Voice chat completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Voice chat error: {str(e)}")
        return jsonify({'error': f'Voice chat failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'agent': os.getenv("AGENT_NAME", "Samantha")})


if __name__ == '__main__':
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not found in environment variables")
        exit(1)
    
    logger.info("🚀 Starting Flask voice chat app...")
    logger.info(f"🤖 Agent: {os.getenv('AGENT_NAME', 'Samantha')}")
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )