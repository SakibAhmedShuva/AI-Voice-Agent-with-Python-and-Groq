import os
import wave
from io import BytesIO
import tempfile
import json

# COMMENTED OUT: Remove gevent for testing to avoid protocol errors
# from gevent import monkey
# monkey.patch_all()
# import gevent

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

class AudioProcessor:
    def __init__(self, ws):
        self.ws = ws
        self.frames = bytearray()
        self.speech_buffer = bytearray()
        self.is_speaking = False
        self.silent_frames_count = 0
        self.min_silent_frames = 15
        self.min_audio_length = SAMPLE_RATE * 0.5  # Minimum 0.5 seconds of audio

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
        if len(self.speech_buffer) > self.min_audio_length:
            # COMMENTED OUT: Remove gevent.spawn for non-gevent setup; use threading if needed
            # gevent.spawn(self.process_and_respond, self.speech_buffer)
            from threading import Thread
            Thread(target=self.process_and_respond, args=(self.speech_buffer,)).start()
        else:
            logger.info("Audio too short, ignoring")
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
            
            # Check if websocket is still open
            try:
                self.ws.send(json.dumps({"type": "user_transcript", "data": transcript}))
            except:
                return  # Connection closed
            
            agent_response = agent.invoke(
                {"messages": [{"role": "user", "content": transcript}]},
                config=agent_config
            )
            response_text = agent_response["messages"][-1].content
            logger.info(f'🤖 Agent response: "{response_text}"')
            
            try:
                self.ws.send(json.dumps({"type": "bot_response_text", "data": response_text}))
            except:
                return  # Connection closed
            
            tts_response = groq_client.audio.speech.create(
                model=os.getenv("TTS_MODEL", "tts-1"),
                voice=os.getenv("TTS_VOICE", "fable"),
                input=response_text,
                response_format="mp3"
            )
            audio_bytes = tts_response.read()
            logger.info("🎵 Speech generated, sending to client.")
            
            try:
                self.ws.send(audio_bytes)
            except:
                return  # Connection closed
                
        except Exception as e:
            logger.error(f"❌ Error in processing thread: {e}")
            try:
                self.ws.send(json.dumps({"type": "error", "data": "Sorry, I encountered an error."}))
            except:
                pass  # Connection closed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'agent': os.getenv("AGENT_NAME", "Samantha")})

@sock.route('/voice')
def voice(ws):
    logger.info("🟢 WebSocket connection established.")
    try:
        # Send confirmation (no delay needed without gevent)
        ws.send(json.dumps({"type": "connected", "data": "Ready for audio"}))  # Send confirmation
        logger.info("Sent connection confirmation to client.")
    except Exception as e:
        logger.error(f"Failed to send connection confirmation: {e}")
        return
    processor = AudioProcessor(ws)
    try:
        while True:
            try:
                data = ws.receive()  # Blocking receive
                if data is None:  # Clean close
                    logger.info("WebSocket closed cleanly by client (None received).")
                    break
                if isinstance(data, bytes):
                    logger.info(f"Received audio chunk: {len(data)} bytes")
                    processor.process_audio(data)
                else:
                    logger.warning(f"Ignored non-binary data: type={type(data)}, content={str(data)[:50]}")
            except ConnectionError as e:
                logger.error(f"Connection closed: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected WS error: {e}")
                break
            # COMMENTED OUT: No gevent.sleep needed in threaded mode
            # gevent.sleep(0.01)
    except Exception as e:
        logger.error(f"🔴 WebSocket error: {e}")
    finally:
        logger.info("⚫ WebSocket connection closed.")

# --- AUDIO CONVERSION FUNCTIONS ---

def convert_webm_to_wav_simple(audio_data):
    """Simple conversion approach - try direct WAV creation first"""
    try:
        # Try to create a minimal WAV file from the raw audio
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)  # 16kHz
            # Skip potential WebM header by trying different offsets
            for offset in [0, 100, 200, 500]:
                if len(audio_data) > offset:
                    try:
                        wf.writeframes(audio_data[offset:])
                        wav_buffer.seek(0)
                        logger.info(f"Successfully created WAV with offset {offset}")
                        return wav_buffer.getvalue()
                    except:
                        wav_buffer.seek(0)
                        continue
        return None
    except Exception as e:
        logger.error(f"Simple conversion failed: {e}")
        return None

def convert_audio_with_fallback(audio_data, filename):
    """Multi-approach audio conversion with fallbacks"""
    logger.info(f"Attempting to convert audio file: {filename}, size: {len(audio_data)} bytes")
    
    # Log preview of incoming data
    logger.info(f"Incoming audio data preview: {audio_data[:50]}")  # First 50 bytes
    
    # Check if audio is too short (less than 0.1 seconds at 16kHz)
    if len(audio_data) < 1600:  # 0.1 seconds * 16000 samples/sec
        # For debugging: Proceed with warning instead of error
        # logger.warning("⚠️ Audio is very short; proceeding anyway for testing.")
        # pass
        # else:
        logger.error("❌ Audio file is too short")
        raise Exception("Audio file is too short. Please record for at least 0.5 seconds.")
    
    # Approach 1: Try direct submission to Groq (sometimes WebM works)
    try:
        logger.info("Trying direct WebM submission to Groq...")
        file_tuple = (filename, audio_data, "audio/webm")
        # Test with a small transcription first
        test_response = groq_client.audio.transcriptions.create(
            file=file_tuple,
            model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )
        logger.info("✅ Direct WebM submission successful!")
        return audio_data, "audio/webm"
    except Exception as e:
        logger.info(f"Direct WebM failed: {e}")
    
    # Approach 2: Try pydub conversion
    try:
        from pydub import AudioSegment
        logger.info("Trying pydub conversion...")
        
        # Write to temp file first
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
            temp_input.write(audio_data)
            temp_input_path = temp_input.name
        
        # Load with pydub (handles various formats)
        audio = AudioSegment.from_file(temp_input_path)
        
        # Convert to WAV format
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_data = wav_buffer.getvalue()
        
        # Cleanup
        os.unlink(temp_input_path)
        
        logger.info("✅ Pydub conversion successful!")
        return wav_data, "audio/wav"
        
    except ImportError:
        logger.warning("❌ pydub not installed. Install with: pip install pydub")
    except Exception as e:
        logger.info(f"Pydub conversion failed: {e}")
    
    # Approach 3: Simple WAV conversion attempt
    try:
        logger.info("Trying simple WAV conversion...")
        wav_data = convert_webm_to_wav_simple(audio_data)
        if wav_data:
            logger.info("✅ Simple WAV conversion successful!")
            return wav_data, "audio/wav"
    except Exception as e:
        logger.info(f"Simple WAV conversion failed: {e}")
    
    # Approach 4: Return original data as MP3 (some WebM files work as MP3)
    try:
        logger.info("Trying as MP3 format...")
        file_tuple = (filename.replace('.webm', '.mp3'), audio_data, "audio/mp3")
        test_response = groq_client.audio.transcriptions.create(
            file=file_tuple,
            model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
            response_format="text",
        )
        logger.info("✅ MP3 format successful!")
        return audio_data, "audio/mp3"
    except Exception as e:
        logger.info(f"MP3 format failed: {e}")
    
    # All approaches failed
    logger.error("❌ All audio conversion approaches failed")
    raise Exception("Could not convert audio to a format compatible with Groq API")

# --- HTTP ENDPOINTS ---

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        logger.info("🎙️ Received audio for transcription (HTTP)")
        
        # Read the audio data
        audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            return jsonify({'error': 'Empty audio file'}), 400
        
        # Convert audio with fallback methods
        converted_data, mime_type = convert_audio_with_fallback(audio_data, audio_file.filename)
        
        # Create proper file tuple for Groq API
        transcription_input = (audio_file.filename, converted_data, mime_type)
        
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

@app.route('/voice-chat-legacy', methods=['POST'])
def voice_chat_legacy():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        logger.info("🎙️ Processing voice chat request (HTTP)")
        
        # Read the audio data
        audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            return jsonify({'error': 'Empty audio file'}), 400
        
        # Convert audio with fallback methods
        converted_data, mime_type = convert_audio_with_fallback(audio_data, audio_file.filename)
        
        # Create proper file tuple for Groq API
        transcription_input = (audio_file.filename, converted_data, mime_type)
        
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
        response.headers['X-Transcript'] = transcript
        
        logger.info("✅ Voice chat completed successfully (HTTP)")
        return response
    except Exception as e:
        logger.error(f"❌ Voice chat error (HTTP): {str(e)}")
        return jsonify({'error': f'Voice chat failed: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.getenv("GROQ_API_KEY"):
        logger.error("❌ GROQ_API_KEY not found in environment variables")
        exit(1)
    
    logger.info("🚀 Starting Flask voice chat app in development mode (no gevent)...")
    
    # Use Flask's built-in server with threading for concurrent handling
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
    logger.info(f"✅ Server running on http://0.0.0.0:{port}")