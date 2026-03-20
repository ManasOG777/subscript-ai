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

# Global job store (single-worker deployment keeps this consistent)
jobs = {}
_lock = threading.Lock()

# Whisper model cache
_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_size):
    with _model_lock:
        if model_size not in _model_cache:
            _model_cache[model_size] = WhisperModel(
                model_size, device='cpu', compute_type='int8'
            )
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


def devanagari_to_hinglish(text):
    """
    Convert Devanagari Hindi text to natural Instagram-style Hinglish (Roman script).

    This ITRANS scheme uses UPPERCASE for long vowels:
      A = aa (long, आ/ा)    I = i (long, ई/ी)    U = u (long, ऊ/ू)
      M = anusvara (nasalization, ं)    H = visarga (ः)

    Example: "Aja Apako batAte haiM eka kahAnI"
    Output:  "Aaj aapako batate hain ek kahani"
    """
    if not text:
        return text
    # Already Roman — return unchanged
    if not any('\u0900' <= c <= '\u097F' for c in text):
        return text
    try:
        roman = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)

        # ── Step 1: Uppercase ITRANS markers — MUST be before lower() ───────
        roman = roman.replace('M', 'n')      # anusvara: haiM→hain
        roman = roman.replace('H', '')       # visarga: silent
        roman = roman.replace('N', 'n')      # retroflex ण → n
        roman = roman.replace('T', 't')      # retroflex ट → t
        roman = roman.replace('D', 'd')      # retroflex ड → d
        roman = roman.replace('R', 'r')      # retroflex ड़ → r
        roman = roman.replace('L', 'l')      # retroflex ळ → l
        roman = roman.replace('.t', 't').replace('.d', 'd')
        roman = roman.replace('.r', 'r').replace('.l', 'l').replace('.h', 'h')
        roman = roman.replace('.n', 'n').replace('.N', 'n')
        # Long vowels → casual Hinglish equivalents
        roman = roman.replace('A', 'aa')     # long aa: batAte→bataate, yAra→yaara
        roman = roman.replace('I', 'i')      # long ii: nahIM→nahin→nahi (after schwa)
        roman = roman.replace('U', 'u')      # long uu
        # Punctuation
        roman = roman.replace('\u0964', '.').replace('\u0965', '.').replace('|', '.')
        roman = re.sub(r'[{}\\\\^~`]', '', roman)

        # ── Step 2: Lowercase ────────────────────────────────────────────────
        roman = roman.lower()

        # ── Step 3: Trailing schwa deletion (MUST be BEFORE middle-aa rule) ──
        # ITRANS adds an inherent 'a' to word-final consonants that is silent in Hindi.
        # e.g. aaja→aaj, eka→ek, yaara→yaar, baata→baat (preserving "baat" ≠ "bata")
        # Must run BEFORE middle-aa to avoid "baata"→"bata" (would strip "baat"'s meaning).
        # Safe: only fires when 'a' is preceded by a consonant (not another vowel),
        # so "kyaa", "meraa" (ending in 'aa') are unaffected here.
        roman = re.sub(r'([^aeiou\s])a(?=\s|$)', r'\1', roman)

        # ── Step 4: Middle 'aa' reduction ───────────────────────────────────
        # consonant + aa + consonant + vowel → consonant + a + consonant + vowel
        # bataate→batate  kahaani→kahani  khaana→khana
        # Preserves: baat (no vowel after final t), yaar (no vowel after final r)
        roman = re.sub(r'([^aeiou\s])aa([^aeiou\s][aeiou])', r'\1a\2', roman)

        # ── Step 5: Word-final 'aa' → 'a' ──────────────────────────────────
        # kyaa→kya  meraa→mera  teraa→tera
        roman = re.sub(r'aa\b', 'a', roman)

        # ── Step 6: Clean up ─────────────────────────────────────────────────
        roman = re.sub(r'\s+', ' ', roman).strip()
        roman = re.sub(r'\s+([.,!?;:])', r'\1', roman)

        # ── Step 7: Capitalize sentence starts ───────────────────────────────
        def cap_sentences(s):
            result = []; cap = True
            for ch in s:
                if cap and ch.isalpha():
                    result.append(ch.upper()); cap = False
                else:
                    result.append(ch)
                if ch in '.!?': cap = True
            return ''.join(result)

        roman = cap_sentences(roman)
        if roman and roman[0].isalpha():
            roman = roman[0].upper() + roman[1:]
        return roman.strip()
    except Exception:
        return text



# ── Translation with retry ─────────────────────────────────────────────────────

def batch_translate_with_retry(texts, translator, max_retries=3):
    """
    Translate a list of texts with batch grouping and retry logic.
    Groups texts into ≤3000-char chunks to stay within API limits.
    Falls back to individual translation if batch split fails.
    Falls back to original text if all retries fail.
    """
    SEPARATOR = ' ||| '
    MAX_CHARS = 3000

    results = list(texts)  # defaults to originals

    # Build batches
    groups = []
    chunk_indices, chunk_buf, chunk_len = [], [], 0

    for i, t in enumerate(texts):
        t = (t or '').strip()
        if not t:
            continue
        needed = len(t) + len(SEPARATOR)
        if chunk_len + needed > MAX_CHARS and chunk_buf:
            groups.append((chunk_indices[:], SEPARATOR.join(chunk_buf)))
            chunk_indices, chunk_buf, chunk_len = [], [], 0
        chunk_indices.append(i)
        chunk_buf.append(t)
        chunk_len += needed

    if chunk_buf:
        groups.append((chunk_indices, SEPARATOR.join(chunk_buf)))

    for indices, joined in groups:
        translated_parts = None

        for attempt in range(max_retries):
            try:
                translated = translator.translate(joined)
                if not translated:
                    raise ValueError("Empty translation response")

                # Try different separator variants (Google Translate may alter spacing)
                parts = None
                for sep in [' ||| ', '||| ', ' |||', '|||', ' | ', '|']:
                    parts = translated.split(sep)
                    if len(parts) == len(indices):
                        break

                if parts and len(parts) == len(indices):
                    translated_parts = [p.strip() for p in parts]
                    break
                else:
                    # Batch split failed → translate individually
                    translated_parts = []
                    for idx in indices:
                        try:
                            single = translator.translate(texts[idx])
                            translated_parts.append((single or texts[idx]).strip())
                        except Exception:
                            translated_parts.append(texts[idx])
                    break

            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))  # exponential backoff
                else:
                    # All retries failed — keep originals
                    translated_parts = [texts[idx] for idx in indices]

        if translated_parts:
            for idx, part in zip(indices, translated_parts):
                results[idx] = part

    return results


# ── Google Drive downloader ────────────────────────────────────────────────────

def gdrive_download(url, dest_path, job_id):
    """Download from Google Drive share link."""
    update_job(job_id, {'progress': 8, 'message': 'Downloading from Google Drive...'})

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
        # ── Step 1: Load model ─────────────────────────────────────────────
        update_job(job_id, {
            'status': 'loading_model', 'progress': 5,
            'message': f'Loading Whisper {model_size} model...'
        })
        model = get_model(model_size)

        # ── Step 2: Extract audio ──────────────────────────────────────────
        update_job(job_id, {
            'status': 'extracting_audio', 'progress': 16,
            'message': 'Extracting audio from video...'
        })

        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_audio.wav")
        try:
            proc = subprocess.run(
                ['ffmpeg', '-i', video_path,
                 '-vn',              # skip video stream
                 '-ar', '16000',     # 16kHz (Whisper requirement)
                 '-ac', '1',         # mono
                 '-c:a', 'pcm_s16le',
                 audio_path, '-y', '-loglevel', 'error'],
                capture_output=True,
                timeout=300
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out. The video may be too large or corrupted.")

        if proc.returncode != 0:
            raise RuntimeError(
                "Could not extract audio from this video. "
                "Please ensure the file contains audio and is not corrupted."
            )

        # ── Step 3: Transcribe ─────────────────────────────────────────────
        update_job(job_id, {
            'status': 'transcribing', 'progress': 32,
            'message': 'Transcribing audio with Whisper AI...'
        })

        lang_arg = None if (not source_lang or source_lang == 'auto') else source_lang

        # Initial prompt improves Hindi/Hinglish accuracy significantly
        initial_prompt = None
        if lang_arg in ('hi', None):
            initial_prompt = (
                "Aaj hum aapko ek interesting story batate hain. "
                "Yeh video bahut acchi hai. Chalte hain shuru karte hain."
            )

        raw_segments, info = model.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
            temperature=0,  # deterministic output
        )
        detected_lang = info.language

        # Collect segments — classify each segment
        # Whisper may output Devanagari (hi), Roman Hinglish (hi romanized), or Latin (en)
        segments = []
        for s in raw_segments:
            text = s.text.strip()
            if not text:
                continue

            is_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
            # Whisper sometimes outputs Roman/Hinglish for Hindi audio
            # Detect via detected_lang or explicit source_lang setting
            is_hindi_roman = (
                not is_devanagari and
                detected_lang == 'hi'
            )

            if is_devanagari:
                # True Devanagari — native Hindi script
                seg_type = 'devanagari'
            elif is_hindi_roman:
                # Whisper output Roman script for Hindi audio
                seg_type = 'hindi_roman'
            else:
                # English or other Latin-script source
                seg_type = 'english'

            segments.append({
                'start': round(s.start, 3),
                'end': round(s.end, 3),
                'text': text,
                'text_en': '' if is_devanagari else text,
                'text_hi': text if is_devanagari else '',
                'text_hinglish': devanagari_to_hinglish(text) if is_devanagari else (text if is_hindi_roman else ''),
                '_seg_type': seg_type,
            })

        if not segments:
            raise RuntimeError(
                "No speech detected in the video. "
                "Please ensure the video has clear audio and try again."
            )

        # ── Step 4: Translate ──────────────────────────────────────────────
        update_job(job_id, {
            'status': 'translating', 'progress': 60,
            'message': f'Translating {len(segments)} segments...'
        })

        devanagari_idx = [i for i, s in enumerate(segments) if s['_seg_type'] == 'devanagari']
        hindi_roman_idx = [i for i, s in enumerate(segments) if s['_seg_type'] == 'hindi_roman']
        english_idx = [i for i, s in enumerate(segments) if s['_seg_type'] == 'english']

        # Devanagari source → translate to English
        if devanagari_idx:
            update_job(job_id, {
                'status': 'translating', 'progress': 65,
                'message': f'Translating Hindi (Devanagari) → English ({len(devanagari_idx)} segments)...'
            })
            hi2en = GoogleTranslator(source='hi', target='en')
            hi_texts = [segments[i]['text_hi'] for i in devanagari_idx]
            en_results = batch_translate_with_retry(hi_texts, hi2en)
            for idx, en_text in zip(devanagari_idx, en_results):
                segments[idx]['text_en'] = en_text

        # Hindi Roman (Hinglish) source:
        # 1. Translate Hinglish → English (Google handles this well)
        # 2. Translate English → Hindi Devanagari
        if hindi_roman_idx:
            update_job(job_id, {
                'status': 'translating', 'progress': 65,
                'message': f'Translating Hinglish → English ({len(hindi_roman_idx)} segments)...'
            })
            hinglish2en = GoogleTranslator(source='auto', target='en')
            hin_texts = [segments[i]['text_hinglish'] for i in hindi_roman_idx]
            en_results = batch_translate_with_retry(hin_texts, hinglish2en)
            for idx, en_text in zip(hindi_roman_idx, en_results):
                segments[idx]['text_en'] = en_text

            # Now translate English → Hindi Devanagari
            update_job(job_id, {
                'status': 'translating', 'progress': 72,
                'message': f'Translating English → Hindi Devanagari ({len(hindi_roman_idx)} segments)...'
            })
            en2hi = GoogleTranslator(source='en', target='hi')
            en_texts = [segments[i]['text_en'] for i in hindi_roman_idx]
            hi_results = batch_translate_with_retry(en_texts, en2hi)
            for idx, hi_text in zip(hindi_roman_idx, hi_results):
                # Only update Devanagari if translation worked (contains Devanagari)
                if any('\u0900' <= c <= '\u097F' for c in hi_text):
                    segments[idx]['text_hi'] = hi_text
                    # Keep original Hinglish for hinglish track (more natural)
                    # text_hinglish already set above

        # English/other source → translate to Hindi
        if english_idx:
            update_job(job_id, {
                'status': 'translating', 'progress': 75,
                'message': f'Translating to Hindi ({len(english_idx)} segments)...'
            })
            en2hi = GoogleTranslator(source='auto', target='hi')
            en_texts = [segments[i]['text_en'] for i in english_idx]
            hi_results = batch_translate_with_retry(en_texts, en2hi)
            for idx, hi_text in zip(english_idx, hi_results):
                segments[idx]['text_hi'] = hi_text
                segments[idx]['text_hinglish'] = devanagari_to_hinglish(hi_text)

        # Remove internal flag
        for s in segments:
            s.pop('_seg_type', None)

        update_job(job_id, {
            'status': 'translating', 'progress': 90,
            'message': 'Translation complete.'
        })

        # ── Step 5: Generate files ─────────────────────────────────────────
        update_job(job_id, {
            'status': 'generating', 'progress': 93,
            'message': 'Writing subtitle files...'
        })

        out_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        os.makedirs(out_dir, exist_ok=True)

        files = {
            'english.srt':           make_srt(segments, 'text_en'),
            'hindi.srt':             make_srt(segments, 'text_hi'),
            'hinglish.srt':          make_srt(segments, 'text_hinglish'),
            'bilingual.srt':         make_bilingual_srt(segments),
            'english.vtt':           make_vtt(segments, 'text_en'),
            'hindi.vtt':             make_vtt(segments, 'text_hi'),
            'hinglish.vtt':          make_vtt(segments, 'text_hinglish'),
            'transcript_en.txt':     '\n'.join(s['text_en'] for s in segments if s.get('text_en')),
            'transcript_hi.txt':     '\n'.join(s['text_hi'] for s in segments if s.get('text_hi')),
            'transcript_hinglish.txt': '\n'.join(s.get('text_hinglish', '') for s in segments),
        }
        for fname, content in files.items():
            with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as f:
                f.write(content)

        with open(os.path.join(out_dir, 'segments.json'), 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        duration = segments[-1]['end'] if segments else 0
        word_count = sum(
            len((s.get('text_en') or s.get('text', '')).split())
            for s in segments
        )

        update_job(job_id, {
            'status': 'done',
            'progress': 100,
            'message': 'Done! Your subtitles are ready.',
            'detected_language': detected_lang,
            'segment_count': len(segments),
            'word_count': word_count,
            'duration': duration,
            'original_name': original_name,
            'segments_preview': segments[:8],
        })

    except Exception as e:
        err_msg = str(e)
        update_job(job_id, {'status': 'error', 'progress': 0, 'message': err_msg})

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

    if model_size not in ('tiny', 'base', 'small', 'medium'):
        model_size = 'base'

    job_id = str(uuid.uuid4())

    if gdrive_url:
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
                run_transcription(job_id, actual_path, model_size, source_lang,
                                  os.path.basename(actual_path))
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
    """SSE stream — sends job updates every 0.7s plus heartbeat every ~15s."""
    def generate():
        heartbeat_counter = 0
        while True:
            data = get_job(job_id)
            # Strip large preview data from stream
            stream_data = {k: v for k, v in data.items() if k != 'segments_preview'}
            yield f"data: {json.dumps(stream_data)}\n\n"
            if data.get('status') in ('done', 'error'):
                break
            time.sleep(0.7)
            heartbeat_counter += 1
            # Heartbeat every ~15s keeps Railway/Nginx/proxy connections alive
            if heartbeat_counter % 21 == 0:
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@app.route('/progress_json/<job_id>')
def progress_json(job_id):
    """
    Fallback polling endpoint for devices where SSE is unreliable (older iOS Safari).
    The frontend polls this every 2s when EventSource fails.
    """
    data = get_job(job_id)
    stream_data = {k: v for k, v in data.items() if k != 'segments_preview'}
    return jsonify(stream_data)


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
        'english_srt':         'english.srt',
        'hindi_srt':           'hindi.srt',
        'hinglish_srt':        'hinglish.srt',
        'bilingual_srt':       'bilingual.srt',
        'english_vtt':         'english.vtt',
        'hindi_vtt':           'hindi.vtt',
        'hinglish_vtt':        'hinglish.vtt',
        'transcript_en':       'transcript_en.txt',
        'transcript_hi':       'transcript_hi.txt',
        'transcript_hinglish': 'transcript_hinglish.txt',
    }
    filename = fmt_map.get(fmt)
    if not filename:
        return jsonify({'error': 'Invalid format'}), 400

    filepath = os.path.join(app.config['OUTPUT_FOLDER'], job_id, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        filepath,
        as_attachment=True,
        download_name=filename,
        mimetype='application/octet-stream',  # force download on all browsers incl. iOS Safari
    )


if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0', threaded=True)
