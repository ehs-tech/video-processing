import os
# Suppress TensorFlow/MediaPipe/TFLite logs BEFORE importing mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TFLITE_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['GLOG_logtostderr'] = '1'

import logging
import subprocess
import json
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
import asyncio
import cv2
import numpy as np
import tempfile
import shutil
import absl.logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time # For potential profiling

# Lower absl verbosity
absl.logging.set_verbosity(absl.logging.ERROR)
# Set up application logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("video_blur")

app = FastAPI(title="Video-Blur & Voice-Change Service with Dynamic Rotation Fix")

import mediapipe as mp
thread_local = threading.local()

# --- CPU and Threading Optimizations ---
num_workers = os.cpu_count() or 2 # Default to 2 if detection fails
cv2.setNumThreads(num_workers)
logger.info(f"OpenCV will use up to {cv2.getNumThreads()} threads.")
logger.info(f"ThreadPoolExecutor will use up to {num_workers} workers.")
# ---

def get_detector():
    if not hasattr(thread_local, 'detector'):
        dp = mp.solutions.face_detection
        thread_local.detector = dp.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    return thread_local.detector

# Locate ffmpeg and ffprobe
FALLBACK_FFMPEG = r"C:\Users\User\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE"
FFMPEG = shutil.which('ffmpeg') or (FALLBACK_FFMPEG if os.path.isfile(FALLBACK_FFMPEG) else None)
FALLBACK_FFPROBE = os.path.join(os.path.dirname(FFMPEG or ''), 'ffprobe.exe')
FFPROBE = shutil.which('ffprobe') or (FALLBACK_FFPROBE if os.path.isfile(FALLBACK_FFPROBE) else None)
if not FFMPEG or not FFPROBE:
    logger.error('ffmpeg/ffprobe not found. Please install and ensure they are on PATH.')
    raise RuntimeError('ffmpeg/ffprobe not found')


def detect_rotation(path: str) -> int:
    """Extract rotation from ffprobe side_data or tags, normalize to [0,360)"""
    cmd = [FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-print_format', 'json', '-show_streams', path]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out)
        stream = data.get('streams', [{}])[0]
        for sd in stream.get('side_data_list', []):
            if 'rotation' in sd:
                return int(sd['rotation']) % 360
        tags = stream.get('tags', {}) or {}
        if 'rotate' in tags:
            return int(round(float(tags['rotate']))) % 360
    except Exception as e:
        logger.warning(f"Could not detect rotation for {path}: {e}")
    return 0


def process_frame(args):
    frame, w, h, blur_scene, rotation = args
    det = get_detector()
    
    if blur_scene:
        # Optimized blur kernel: less aggressive, capped, ensure odd
        k = min(max(w,h)//15, 51) 
        if not k%2: k+=1
        processed = cv2.blur(frame, (k, k))
    else:
        small = cv2.resize(frame, (w//2, h//2)) # Consider w//3, h//3 for more speed
        results = det.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        mask = np.zeros((h, w), dtype=np.uint8)
        if results.detections:
            for d in results.detections:
                r = d.location_data.relative_bounding_box
                x1,y1 = int(r.xmin*w), int(r.ymin*h)
                bw, bh = int(r.width*w), int(r.height*h)
                pw, ph = int(bw*0.6), int(bh*0.3) # Padding for blur area
                x0, y0 = max(0, x1-pw), max(0, y1-ph)
                x2, y2 = min(w, x1+bw+pw), min(h, y1+bh+ph)
                mask[y0:y2, x0:x2] = 255
        
        # Optimized dilate kernel: smaller than original (21,81)
        # Tune (e.g., (7,21), (9,31)) based on visual needs and performance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,41)) 
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Optimized blur kernel
        k = min(max(w,h)//15, 51) 
        if not k%2: k+=1
        blurred = cv2.blur(frame, (k, k))
        
        fg = cv2.bitwise_and(blurred, blurred, mask=mask)
        bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        processed = cv2.add(fg, bg)
    
    fix = (360 - rotation) % 360
    if fix == 90:
        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
    elif fix == 180:
        processed = cv2.rotate(processed, cv2.ROTATE_180)
    elif fix == 270:
        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return processed.tobytes()

# --- Generator for streaming frames ---
def frame_generator_for_processing(video_path: str, frame_w: int, frame_h: int, blur_scene_flag: bool, video_rotation: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video for streaming: {video_path}")
        raise RuntimeError(f'Cannot open input for streaming: {video_path}')
    
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        yield (frame, frame_w, frame_h, blur_scene_flag, video_rotation)
    cap.release()
    logger.info(f"Streamed {frame_count} frames for processing.")
# ---

def heavy_process(path: str, blur_scene: bool, anonymize_voice: bool) -> str:
    logger.info('=== START heavy_process: blur_scene=%s, anonymize_voice=%s for %s', blur_scene, anonymize_voice, os.path.basename(path))
    process_start_time = time.perf_counter()

    rot = detect_rotation(path)
    logger.info('Detected metadata rotation: %d°', rot)
    
    # Capture initial video properties
    cap_meta = cv2.VideoCapture(path)
    if not cap_meta.isOpened(): raise RuntimeError(f'Cannot open input for metadata: {path}')
    fps = cap_meta.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Handle cases where FPS might not be read correctly
        logger.warning("FPS read as 0, defaulting to 25. FFprobe might be needed for accurate FPS.")
        fps = 25.0 
    w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()

    if w == 0 or h == 0:
        raise RuntimeError(f"Could not read frame dimensions for {path}")

    logger.info(f"Input video properties: {w}x{h} @ {fps:.2f} FPS")

    ow, oh = (h, w) if rot in (90, 270) else (w, h)
    vid_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Ensure fps is a string representation of an integer if it's a whole number, or float otherwise
    fps_str = str(int(fps)) if fps.is_integer() else str(fps)

    cmd_ffmpeg_video = [FFMPEG,'-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{ow}x{oh}','-r',fps_str,'-i','pipe:0',
                       '-c:v','libx264','-preset','ultrafast','-crf','28','-pix_fmt','yuv420p', vid_tmp]
    
    ffmpeg_proc = subprocess.Popen(cmd_ffmpeg_video, stdin=subprocess.PIPE)
    
    logger.info(f'Processing frames using {num_workers} workers (from frame_generator)...')
    
    frames_processed_count = 0
    frame_read_time = 0
    frame_process_time = 0
    pipe_write_time = 0

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        # Create the generator instance
        gen = frame_generator_for_processing(path, w, h, blur_scene, rot)
        for processed_frame_bytes in ex.map(process_frame, gen):
            t_pipe_start = time.perf_counter()
            ffmpeg_proc.stdin.write(processed_frame_bytes)
            pipe_write_time += (time.perf_counter() - t_pipe_start)
            frames_processed_count +=1

    if ffmpeg_proc.stdin:
        ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    
    video_processing_duration = time.perf_counter() - process_start_time
    logger.info(f'Video frames ({frames_processed_count}) processed and written to {vid_tmp} in {video_processing_duration:.2f}s.')
    if frames_processed_count > 0:
        logger.info(f"Average pipe write time per frame: {pipe_write_time / frames_processed_count:.4f}s")


    # Audio processing
    audio_start_time = time.perf_counter()
    aud_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.m4a').name
    if anonymize_voice:
        logger.info('Changing voice pitch')
        fstr='asetrate=44100*0.75,aresample=44100,atempo=1.333' # Deeper pitch shift for all voices
        subprocess.run([FFMPEG,'-y','-i',path,'-vn','-af',fstr,'-c:a','aac','-b:a','128k',aud_tmp], check=True, capture_output=True) # Added capture_output for debugging
    else:
        subprocess.run([FFMPEG,'-y','-i',path,'-vn','-c:a','copy',aud_tmp], check=True, capture_output=True)
    logger.info(f'Audio processing took {time.perf_counter() - audio_start_time:.2f}s. Audio temp: {aud_tmp}')

    # Muxing video and audio
    mux_start_time = time.perf_counter()
    final_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{os.path.basename(path)}").name # Ensure unique suffix
    subprocess.run([FFMPEG,'-y','-i',vid_tmp,'-i',aud_tmp,'-c:v','copy','-c:a','copy',final_output_path], check=True, capture_output=True) # Use copy for audio if already AAC
    logger.info(f'Muxing took {time.perf_counter() - mux_start_time:.2f}s. Final output: {final_output_path}')
    
    # Clean up temporary files
    try:
        os.remove(vid_tmp)
        os.remove(aud_tmp)
    except OSError as e:
        logger.warning(f"Error removing temp files: {e}")

    total_duration = time.perf_counter() - process_start_time
    logger.info(f'=== FINISHED heavy_process for {os.path.basename(path)} in {total_duration:.2f}s ===')
    return final_output_path

@app.get('/')
async def root_get_slash(): # Renamed to avoid conflict
    return {"status": "ok /"}

@app.get('') # This might conflict if deployed at root, usually a path like '/status' is better
async def root_get_empty(): # Renamed
    return {"status": "ok empty"}

@app.post('/blur_video')
async def blur_video(file: UploadFile=File(...), blur_scene:bool=Query(False), anonymize_voice:bool=Query(False)):
    logger.info('/blur_video called; blur_scene=%s anonymize_voice=%s filename=%s', blur_scene, anonymize_voice, file.filename)
    
    # Save uploaded file to a temporary path
    tmp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]).name
    try:
        with open(tmp_input_path,'wb') as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded file saved to temporary path: {tmp_input_path}")
    except Exception as e:
        logger.exception(f"Failed to save uploaded file: {file.filename}")
        # Clean up if saving failed before processing starts
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {str(e)}")
    finally:
        await file.close() # Ensure file is closed

    loop = asyncio.get_running_loop()
    output_file_path = None
    try:
        output_file_path = await loop.run_in_executor(None, heavy_process, tmp_input_path, blur_scene, anonymize_voice)
    except Exception as e:
        logger.exception(f'Error processing video {file.filename}')
        raise HTTPException(status_code=500, detail=f"Error during video processing: {str(e)}")
    finally:
        # Clean up the temporary input file after processing
        if os.path.exists(tmp_input_path):
            try:
                os.remove(tmp_input_path)
                logger.info(f"Cleaned up temporary input file: {tmp_input_path}")
            except OSError as e:
                logger.warning(f"Could not remove temporary input file {tmp_input_path}: {e}")
    
    if not output_file_path or not os.path.exists(output_file_path):
         raise HTTPException(status_code=500, detail="Processing finished but output file not found.")

    return FileResponse(output_file_path, media_type='video/mp4', filename=os.path.basename(output_file_path))

@app.post('/echo_video')
async def echo_video(file: UploadFile = File(...)):
    """Receives a video file and returns it unchanged."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]).name
    try:
        with open(tmp, 'wb') as f:
            shutil.copyfileobj(file.file, f) # Use shutil.copyfileobj for UploadFile
    except Exception as e:
        logger.error(f"Could not echo video: {e}")
        if os.path.exists(tmp): os.remove(tmp) # Clean up temp file on error
        raise HTTPException(status_code=500, detail="Could not process echo video request.")
    finally:
        await file.close()

    return FileResponse(tmp, media_type='video/mp4', filename=os.path.basename(tmp))

@app.get('/health')
async def health(): return {'status':'ok', 'message': 'Video processing service is running'}
