import os
import wave
from io import BytesIO
import threading
import tempfile

from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request, send_file, Response
from flask_sock import Sock
from groq import Groq
from loguru import logger
import webrtcvad

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

# Initialize Flask app and WebSocket
app = Flask(__name__)
sock = Sock(app)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configure VAD
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 3))
vad = webrtcvad.Vad()
vad.set_mode(VAD_AGGRESSIVENESS)

# Constants for audio processing
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# --- WebSocket Class for Continuous Conversation ---
class AudioProcessor:
    def __init__(self, ws):
        self.ws = ws
        self.frames = bytearray()
        self.speech_buffer = bytearray()
        self.is_speaking = False
        self.silent_frames_count = 0
        self.min_silent_frames = 15

    def process_audio(self, audio_chunk):
        self.frames.extend(audio_chunk)
        
        while len(self.frames) >= FRAME_SIZE:
            frame = self.frames[:FRAME_SIZE]
            del self.frames[:FRAME_SIZE]
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                self.speech_buffer.extend(frame)
                self.is_speaking = True
                self.silent_frames_count = 0
            elif self.is_speaking:
                self.speech_buffer.extend(frame)
                self.silent_frames_count += 1
                if self.silent_frames_count > self.min_silent_frames:
                    self.end_of_speech()
    
    def end_of_speech(self):
        logger.info("🎙️ End of speech detected.")
        self.is_speaking = False
        self.silent_frames_count = 0
        if len(self.speech_buffer) > SAMPLE_RATE:
            threading.Thread(target=self.process_and_respond, args=(self.speech_buffer,)).start()
        self.speech_buffer = bytearray()

    def process_and_respond(self, audio_data):
        try:
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
            wav_buffer.seek(0)
            
            file_tuple = ("user_speech.wav", wav_buffer.read(), "audio/wav")
            transcript = groq_client.audio.transcriptions.create(
                file=file_tuple,
                model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
                response_format="text",
            )
            logger.info(f'👂 Transcribed: "{transcript}"')
            self.ws.send(f'{{"type": "user_transcript", "data": "{transcript}"}}')

            agent_response = agent.invoke(
                {"messages": [{"role": "user", "content": transcript}]},
                config=agent_config
            )
            response_text = agent_response["messages"][-1].content
            logger.info(f'🤖 Agent response: "{response_text}"')
            self.ws.send(f'{{"type": "bot_response_text", "data": "{response_text}"}}')
            
            tts_response = groq_client.audio.speech.create(
                model=os.getenv("TTS_MODEL", "tts-1"),
                voice=os.getenv("TTS_VOICE", "fable"),
                input=response_text,
                response_format="mp3"
            )
            audio_bytes = tts_response.read()
            logger.info("🎵 Speech generated, sending to client.")
            self.ws.send(audio_bytes)
        except Exception as e:
            logger.error(f"❌ Error in processing thread: {e}")
            self.ws.send(f'{{"type": "error", "data": "Sorry, I encountered an error."}}')

# --- Main App Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'agent': os.getenv("AGENT_NAME", "Samantha")})

# --- WebSocket Route for Continuous Call Mode ---
@sock.route('/voice')
def voice(ws):
    logger.info("🟢 WebSocket connection established.")
    processor = AudioProcessor(ws)
    try:
        while True:
            data = ws.receive()
            if isinstance(data, bytes):
                processor.process_audio(data)
    except Exception as e:
        logger.error(f"🔴 WebSocket error: {e}")
    finally:
        logger.info("⚫ WebSocket connection closed.")

# --- RESTORED: Original HTTP Endpoints for Push-to-Talk/Text Mode ---

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        logger.info("🎙️ Received audio for transcription (HTTP)")
        
        transcription_input = (audio_file.filename, audio_file.read())
        
        transcript = groq_client.audio.transcriptions.create(
            file=transcription_input,
            model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )
        
        logger.info(f'👂 Transcribed (HTTP): "{transcript}"')
        return jsonify({'transcript': transcript})
    except Exception as e:
        logger.error(f"❌ Transcription error (HTTP): {str(e)}")
        return jsonify({'error': f'Transcription failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f'💬 Processing message (HTTP): "{message}"')
        
        agent_response = agent.invoke(
            {"messages": [{"role": "user", "content": message}]}, 
            config=agent_config
        )
        response_text = agent_response["messages"][-1].content
        
        logger.info(f'🤖 Agent response (HTTP): "{response_text}"')
        return jsonify({'response': response_text})
    except Exception as e:
        logger.error(f"❌ Chat error (HTTP): {str(e)}")
        return jsonify({'error': f'Chat processing failed: {str(e)}'}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        logger.info(f'🔊 Synthesizing speech (HTTP) for: "{text}"')
        
        tts_response = groq_client.audio.speech.create(
            model=os.getenv("TTS_MODEL", "tts-1"),
            voice=os.getenv("TTS_VOICE", "fable"),
            response_format="mp3",
            input=text,
        )
        
        audio_bytes = tts_response.read()
        logger.info("🎵 Speech generated successfully (HTTP)")
        
        return send_file(BytesIO(audio_bytes), mimetype='audio/mpeg', as_attachment=False)
    except Exception as e:
        logger.error(f"❌ TTS error (HTTP): {str(e)}")
        return jsonify({'error': f'Speech synthesis failed: {str(e)}'}), 500

# This combined endpoint is for the push-to-talk mode
@app.route('/voice-chat-legacy', methods=['POST'])
def voice_chat_legacy():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        logger.info("🎙️ Processing voice chat request (HTTP)")
        
        transcription_input = (audio_file.filename, audio_file.read())
        
        transcript = groq_client.audio.transcriptions.create(
            file=transcription_input,
            model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )
        logger.info(f'👂 Transcribed (HTTP): "{transcript}"')
        
        agent_response = agent.invoke(
            {"messages": [{"role": "user", "content": transcript}]}, 
            config=agent_config
        )
        response_text = agent_response["messages"][-1].content
        logger.info(f'🤖 Agent response (HTTP): "{response_text}"')
        
        tts_response = groq_client.audio.speech.create(
            model=os.getenv("TTS_MODEL", "tts-1"),
            voice=os.getenv("TTS_VOICE", "fable"),
            response_format="mp3",
            input=response_text,
        )
        
        audio_bytes = tts_response.read()
        
        response = Response(BytesIO(audio_bytes), mimetype='audio/mpeg')
        response.headers['X-Transcript'] = transcript # Send back transcript for UI
        
        logger.info("✅ Voice chat completed successfully (HTTP)")
        return response
    except Exception as e:
        logger.error(f"❌ Voice chat error (HTTP): {str(e)}")
        return jsonify({'error': f'Voice chat failed: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not found in environment variables")
        exit(1)
    
    logger.info("🚀 Starting Flask voice chat app in combined mode...")
    
    try:
        from gevent import pywsgi
        from geventwebsocket.handler import WebSocketHandler
        port = int(os.getenv('PORT', 5000))
        server = pywsgi.WSGIServer(('0.0.0.0', port), app, handler_class=WebSocketHandler)
        logger.info(f"✅ Server running on http://0.0.0.0:{port}")
        server.serve_forever()
    except ImportError:
        logger.error("❌ gevent and gevent-websocket are not installed.")
        logger.error("Please run 'pip install gevent gevent-websocket' for WebSocket support.")
        exit(1)