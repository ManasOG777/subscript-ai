import os
import re
import json
import uuid
import threading
import time
import subprocess
import requests
from flask import Flask, render_template, request, jsonify, Response, send_file, stream_with_context
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(BASE_DIR, 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4 GB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global job store
jobs = {}
_lock = threading.Lock()

# Whisper model cache (avoid reloading for each job)
_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_size):
    with _model_lock:
        if model_size not in _model_cache:
            # int8 quantization = 4× faster on CPU with no quality loss
            _model_cache[model_size] = WhisperModel(model_size, device='cpu', compute_type='int8')
        return _model_cache[model_size]


def set_job(job_id, data):
    with _lock:
        jobs[job_id] = data


def update_job(job_id, updates):
    with _lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def get_job(job_id):
    with _lock:
        return dict(jobs.get(job_id, {}))


# ── Time formatters ────────────────────────────────────────────────────────────

def to_srt_time(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def to_vtt_time(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


# ── Subtitle generators ────────────────────────────────────────────────────────

def make_srt(segments, key):
    out, idx = [], 1
    for seg in segments:
        text = seg.get(key, '').strip()
        if text:
            out.append(f"{idx}\n{to_srt_time(seg['start'])} --> {to_srt_time(seg['end'])}\n{text}\n")
            idx += 1
    return "\n".join(out)


def make_vtt(segments, key):
    out = ["WEBVTT\n"]
    for seg in segments:
        text = seg.get(key, '').strip()
        if text:
            out.append(f"{to_vtt_time(seg['start'])} --> {to_vtt_time(seg['end'])}\n{text}\n")
    return "\n".join(out)


def devanagari_to_hinglish(text):
    """Transliterate Devanagari → casual Roman (ITRANS scheme, then clean up)."""
    if not text:
        return text
    try:
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        # Light cleanup: collapse double chars for readability (aa→a, ii→i, uu→u at word ends)
        roman = re.sub(r'\baa\b', 'a', roman)
        roman = roman.replace('aa', 'aa').replace('ii', 'ee').replace('uu', 'oo')
        roman = re.sub(r'([A-Z])', lambda m: m.group(1).lower(), roman)
        return roman.strip()
    except Exception:
        return text


def make_bilingual_srt(segments):
    out, idx = [], 1
    for seg in segments:
        en = seg.get('text_en', '').strip()
        hi = seg.get('text_hi', '').strip()
        if en or hi:
            text = f"{en}\n{hi}" if (en and hi) else (en or hi)
            out.append(f"{idx}\n{to_srt_time(seg['start'])} --> {to_srt_time(seg['end'])}\n{text}\n")
            idx += 1
    return "\n".join(out)


# ── Google Drive downloader ────────────────────────────────────────────────────

def gdrive_download(url, dest_path, job_id):
    """Download from Google Drive share link or direct link."""
    update_job(job_id, {'progress': 8, 'message': 'Downloading from Google Drive...'})

    # Extract file ID
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/open\?id=([a-zA-Z0-9_-]+)',
    ]
    file_id = None
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            file_id = m.group(1)
            break

    if not file_id:
        raise ValueError("Could not extract Google Drive file ID from URL.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()
    response = session.get(download_url, stream=True)

    # Handle virus-scan warning page
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(download_url, stream=True)
            break

    if response.status_code != 200:
        raise ValueError(f"Google Drive download failed (HTTP {response.status_code}). Make sure the file is publicly shared.")

    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        raise ValueError("Google Drive returned an HTML page. The file may not be publicly shared or requires sign-in.")

    # Get filename from content-disposition if available
    cd = response.headers.get('content-disposition', '')
    fname_match = re.search(r'filename="?([^";\n]+)"?', cd)
    if fname_match:
        orig_name = fname_match.group(1).strip()
        ext = os.path.splitext(orig_name)[1].lower() or '.mp4'
        dest_path = os.path.splitext(dest_path)[0] + ext

    total = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = int(8 + (downloaded / total) * 12)
                    update_job(job_id, {
                        'progress': pct,
                        'message': f'Downloading... {downloaded // (1024*1024)} MB / {total // (1024*1024)} MB'
                    })

    return dest_path


# ── Core transcription worker ──────────────────────────────────────────────────

def run_transcription(job_id, video_path, model_size, source_lang, original_name):
    try:
        update_job(job_id, {'status': 'loading_model', 'progress': 5, 'message': f'Loading Whisper {model_size} model...'})
        model = get_model(model_size)

        update_job(job_id, {'status': 'extracting_audio', 'progress': 16, 'message': 'Extracting audio...'})

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_audio.wav")
        proc = subprocess.run(
            ['ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1',
             '-c:a', 'pcm_s16le', audio_path, '-y', '-loglevel', 'quiet'],
            capture_output=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {proc.stderr.decode()[:300]}")

        update_job(job_id, {'status': 'transcribing', 'progress': 32, 'message': 'Transcribing with Whisper AI...'})

        lang_arg = None if (not source_lang or source_lang == 'auto') else source_lang
        raw_segments, info = model.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,           # skip silence → faster
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        detected_lang = info.language

        segments = []
        for s in raw_segments:
            text = s.text.strip()
            if text:
                segments.append({
                    'start': s.start,
                    'end': s.end,
                    'text': text,
                    'text_en': text,
                    'text_hi': '',
                    'text_hinglish': '',
                })

        update_job(job_id, {'status': 'translating', 'progress': 62, 'message': 'Translating to Hindi...'})

        # Batch translate: join all text with a sentinel, one API call per ~4000 chars
        SEPARATOR = ' ||| '
        MAX_CHARS = 4000
        translator = GoogleTranslator(source='auto', target='hi')

        def batch_translate(texts):
            """Split into chunks ≤ MAX_CHARS, translate each chunk, reassemble."""
            results = [''] * len(texts)
            chunk_indices, chunk_buf, chunk_len = [], [], 0
            groups = []  # list of (indices, joined_text)

            for i, t in enumerate(texts):
                if chunk_len + len(t) + len(SEPARATOR) > MAX_CHARS and chunk_buf:
                    groups.append((chunk_indices[:], SEPARATOR.join(chunk_buf)))
                    chunk_indices, chunk_buf, chunk_len = [], [], 0
                chunk_indices.append(i)
                chunk_buf.append(t)
                chunk_len += len(t) + len(SEPARATOR)

            if chunk_buf:
                groups.append((chunk_indices, SEPARATOR.join(chunk_buf)))

            for indices, joined in groups:
                try:
                    translated = translator.translate(joined) or joined
                    parts = translated.split(SEPARATOR.strip())
                    # Pad/trim to match original count
                    while len(parts) < len(indices):
                        parts.append(parts[-1] if parts else '')
                    for idx, part in zip(indices, parts):
                        results[idx] = part.strip()
                except Exception:
                    for idx in indices:
                        results[idx] = texts[idx]

            return results

        total = len(segments)
        update_job(job_id, {'status': 'translating', 'progress': 65,
                             'message': f'Batch translating {total} segments...'})

        en_texts = [s['text_en'] for s in segments]
        hi_texts = batch_translate(en_texts)

        for i, hi in enumerate(hi_texts):
            segments[i]['text_hi'] = hi
            segments[i]['text_hinglish'] = devanagari_to_hinglish(hi)

        update_job(job_id, {'status': 'translating', 'progress': 90,
                             'message': 'Translation complete.'})

        update_job(job_id, {'status': 'generating', 'progress': 93, 'message': 'Writing subtitle files...'})

        out_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(out_dir, exist_ok=True)

        files = {
            'english.srt': make_srt(segments, 'text_en'),
            'hindi.srt': make_srt(segments, 'text_hi'),
            'hinglish.srt': make_srt(segments, 'text_hinglish'),
            'bilingual.srt': make_bilingual_srt(segments),
            'english.vtt': make_vtt(segments, 'text_en'),
            'hindi.vtt': make_vtt(segments, 'text_hi'),
            'hinglish.vtt': make_vtt(segments, 'text_hinglish'),
            'transcript_en.txt': '\n'.join(s['text_en'] for s in segments),
            'transcript_hi.txt': '\n'.join(s['text_hi'] for s in segments),
            'transcript_hinglish.txt': '\n'.join(s.get('text_hinglish', '') for s in segments),
        }
        for fname, content in files.items():
            with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
                f.write(content)

        with open(os.path.join(out_dir, 'segments.json'), 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        duration = segments[-1]['end'] if segments else 0
        word_count = sum(len(s['text_en'].split()) for s in segments)

        update_job(job_id, {
            'status': 'done',
            'progress': 100,
            'message': 'Done!',
            'detected_language': detected_lang,
            'segment_count': len(segments),
            'word_count': word_count,
            'duration': duration,
            'original_name': original_name,
            'segments_preview': segments[:8],
        })

    except Exception as e:
        update_job(job_id, {'status': 'error', 'progress': 0, 'message': str(e)})

    finally:
        for p in [video_path, os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_audio.wav")]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    model_size = request.form.get('model', 'base')
    source_lang = request.form.get('language', 'auto')
    gdrive_url = request.form.get('gdrive_url', '').strip()

    valid_models = ['tiny', 'base', 'small', 'medium']
    if model_size not in valid_models:
        model_size = 'base'

    job_id = str(uuid.uuid4())
    original_name = 'video'

    if gdrive_url:
        # Google Drive path
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.mp4")
        set_job(job_id, {
            'status': 'downloading',
            'progress': 2,
            'message': 'Preparing Google Drive download...',
            'original_name': gdrive_url[:60],
        })

        def gdrive_worker():
            try:
                actual_path = gdrive_download(gdrive_url, video_path, job_id)
                run_transcription(job_id, actual_path, model_size, source_lang, os.path.basename(actual_path))
            except Exception as e:
                update_job(job_id, {'status': 'error', 'progress': 0, 'message': str(e)})

        threading.Thread(target=gdrive_worker, daemon=True).start()

    elif 'video' in request.files:
        file = request.files['video']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        original_name = file.filename
        ext = os.path.splitext(original_name)[1].lower() or '.mp4'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}{ext}")
        file.save(video_path)

        set_job(job_id, {
            'status': 'queued',
            'progress': 2,
            'message': 'Queued for processing...',
            'original_name': original_name,
        })

        threading.Thread(
            target=run_transcription,
            args=(job_id, video_path, model_size, source_lang, original_name),
            daemon=True
        ).start()

    else:
        return jsonify({'error': 'No video file or Google Drive URL provided'}), 400

    return jsonify({'job_id': job_id})


@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    """Handle multiple file uploads — returns list of job_ids."""
    model_size = request.form.get('model', 'base')
    source_lang = request.form.get('language', 'auto')
    files = request.files.getlist('videos')

    if not files:
        return jsonify({'error': 'No files provided'}), 400

    job_ids = []
    for file in files:
        if not file or file.filename == '':
            continue
        job_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower() or '.mp4'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}{ext}")
        file.save(video_path)

        set_job(job_id, {
            'status': 'queued',
            'progress': 2,
            'message': 'Queued...',
            'original_name': file.filename,
        })

        threading.Thread(
            target=run_transcription,
            args=(job_id, video_path, model_size, source_lang, file.filename),
            daemon=True
        ).start()

        job_ids.append({'job_id': job_id, 'name': file.filename})

    return jsonify({'jobs': job_ids})


@app.route('/progress/<job_id>')
def progress(job_id):
    def generate():
        while True:
            data = get_job(job_id)
            # Don't send segments_preview in SSE stream (too large)
            stream_data = {k: v for k, v in data.items() if k != 'segments_preview'}
            yield f"data: {json.dumps(stream_data)}\n\n"
            if data.get('status') in ('done', 'error'):
                break
            time.sleep(0.7)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/segments/<job_id>')
def segments(job_id):
    path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, 'segments.json')
    if not os.path.exists(path):
        return jsonify({'error': 'Not found'}), 404
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))


@app.route('/download/<job_id>/<fmt>')
def download(job_id, fmt):
    fmt_map = {
        'english_srt': 'english.srt',
        'hindi_srt': 'hindi.srt',
        'hinglish_srt': 'hinglish.srt',
        'bilingual_srt': 'bilingual.srt',
        'english_vtt': 'english.vtt',
        'hindi_vtt': 'hindi.vtt',
        'hinglish_vtt': 'hinglish.vtt',
        'transcript_en': 'transcript_en.txt',
        'transcript_hi': 'transcript_hi.txt',
        'transcript_hinglish': 'transcript_hinglish.txt',
    }
    filename = fmt_map.get(fmt)
    if not filename:
        return jsonify({'error': 'Invalid format'}), 400

    filepath = os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    return send_file(filepath, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0', threaded=True)
