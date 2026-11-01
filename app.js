/**
 * app.js — Kokoro TTS client‑side initialization, device detection,
 * model loading, text‑to‑speech generation, audio playback and download.
 *
 * This module is loaded as an ES module from index.html via:
 *   <script type="module" src="app.js"></script>
 *
 * Kokoro‑JS is imported from the jsDelivr CDN (no backend, no API key).
 */
import { KokoroTTS } from 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm';

const MODELS_IDB_KEY = 'kokoro-models';
const DEFAULT_VOICE = 'am_adam';
const SAMPLE_RATE = 24000; // Kokoro‑82M output sample rate

/* -------------------------------------------------------------------------
   Helper: create a WAV Blob from Float32Array (interleaved, mono, 24kHz)
   ------------------------------------------------------------------------- */
function float32ToWavPCM16(float32Arr, sampleRate = SAMPLE_RATE) {
  const len = float32Arr.length;
  const headerLen = 44;
  const buffer = new ArrayBuffer(headerLen + len * 2);
  const view = new DataView(buffer);

  // Helper to write string to DataView
  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // RIFF chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + len * 2, true);
  writeString(view, 8, 'WAVE');

  // fmt sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true); // audioFormat (PCM)
  view.setUint16(22, 1, true); // num channels (mono)
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byteRate
  view.setUint16(32, 2, true); // blockAlign
  view.setUint16(34, 16, true); // bitsPerSample

  // data sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, len * 2, true);

  // Write PCM samples
  let offset = headerLen;
  for (let i = 0; i < len; i++) {
    const s = Math.max(-1, Math.min(1, float32Arr[i]));
    const sample = s < 0 ? s * 0x8000 : s * 0x7FFF;
    view.setInt16(offset, sample, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/* -------------------------------------------------------------------------
   Helper: format bytes / progress
   ------------------------------------------------------------------------- */
function formatProgress(downloaded, total) {
  if (total === 0) return '0%';
  const pct = Math.round((downloaded / total) * 100);
  return `${pct}%`;
}

/* -------------------------------------------------------------------------
   Device detection: prefers WebGPU, falls back to WASM
   ------------------------------------------------------------------------- */
async function detectDevice() {
  // Prefer WebGPU if available and functional
  if (navigator.gpu && navigator.gpu.requestAdapter) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        // WebGPU is available; we will use fp32 precision
        return { mode: 'webgpu', device: 'webgpu' };
      }
    } catch (e) {
      // WebGPU request failed, fall through to WASM
      console.warn('WebGPU request failed, falling back to WASM:', e);
    }
  }
  // Fallback to WASM with quantized model (q4) to keep memory ~80 MB
  return { mode: 'wasm', device: 'wasm' };
}

/* -------------------------------------------------------------------------
   Initialize Kokoro TTS (via the CDN bundle)
   ------------------------------------------------------------------------- */
async function initKokoro({mode, device}) {
  // The CDN bundle exports: KokoroTTS, TextSplitterStream, env
  // We instantiate with the selected precision and device.
  // dtype 'q4' = quantized, 'fp32' = full precision.
  const dtype = mode === 'webgpu' ? 'fp32' : 'q4';

  window.statusBar = document.getElementById('status');
  window.modeText = document.getElementById('modeText');
  window.progress = document.getElementById('progress');
  window.audioWrapper = document.getElementById('audioWrapper');
  window.audioPlayer = document.getElementById('audioPlayer');
  window.downloadBtn = document.getElementById('downloadBtn');
  window.generateBtn = document.getElementById('generateBtn');
  window.clearBtn = document.getElementById('clearBtn');
  window.textInput = document.getElementById('textInput');
  window.voiceSelect = document.getElementById('voiceSelect');
  window.voiceInfo = document.getElementById('voiceInfo');
  window.charCount = document.getElementById('charCount');

  // Update status to show initialization
  updateMode('Initializing Kokoro…');

  // Load the model from Hugging Face (onnx‑community Kokoro‑82M)
  // The from_pretrained method will download & cache the model.
  // We pass progress_callback to update UI.
  const progress = { loaded: 0, total: 100 };

  const tts = await KokoroTTS.from_pretrained(
    'onnx-community/Kokoro-82M-v1.0-ONNX',
    {
      dtype,
      device,
      progress_callback: (p) => {
        // p is a percentage 0‑100
        progress.loaded = p;
        updateProgress(p);
      },
    }
  );
  // Attach to window for legacy event handlers that reference window.tts
  window.tts = tts;

  // Populate voice dropdown with known Kokoro voices (from the library's $ map)
  const voiceMap = window.tts.voices; // Map of voiceName -> voiceInfo
  const voiceNames = Object.keys(voiceMap);
  window.voiceSelect.innerHTML = ''; // clear default
  voiceNames.forEach((v) => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    if (v === DEFAULT_VOICE) opt.selected = true;
    window.voiceSelect.appendChild(opt);
  });

  // Show default voice info
  updateVoiceInfo(DEFAULT_VOICE);

  // Store model info in IndexedDB for future instant loads (simple metadata)
  try {
    const db = await openIDB();
    await putIDB(db, 'model-state', { mode, device, dtype, loaded: progress.loaded });
    await closeIDB(db);
  } catch (e) {
    console.info('IndexedDB not available or error:', e);
  }

  // Set final mode display
  updateMode(`Mode: ${mode.toUpperCase()}`);
  updateProgress(100);

  // Enable UI
  window.generateBtn.classList.remove('btn-loading');
  window.generateBtn.disabled = false;
}

/* -------------------------------------------------------------------------
   UI update helpers
   ------------------------------------------------------------------------- */
function updateMode(text) {
  if (window.modeText) window.modeText.textContent = text;
}
function updateProgress(pct) {
  if (window.progress) window.progress.textContent = `${pct}%`;
}
function updateVoiceInfo(voiceName) {
  if (!window.voiceInfo) return;
  const voiceMap = window.tts?.voices || {};
  const info = voiceMap[voiceName];
  if (info) {
    window.voiceInfo.textContent =
      `${info.name} • ${info.gender} • ${info.language}`;
  } else {
    window.voiceInfo.textContent = 'Voice info unavailable';
  }
}

/* -------------------------------------------------------------------------
   IndexedDB lite helpers (open / put / get / close)
   ------------------------------------------------------------------------- */
function openIDB() {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(MODELS_IDB_KEY, 1);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains('model-state')) {
        db.createObjectStore('model-state');
      }
    };
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
}
function putIDB(db, store, data) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite');
    const storeObj = tx.objectStore(store);
    storeObj.put(data, Date.now().toString());
    tx.oncomplete = () => resolve();
    tx.onerror = (e) => reject(e.target.error);
  });
}
function closeIDB(db) {
  db.close();
}

/* -------------------------------------------------------------------------
   Event wiring
   ------------------------------------------------------------------------- */
async function onGenerate() {
  const text = window.textInput.value.trim();
  if (!text) {
    alert('Please enter some text.');
    return;
  }

  const voice = window.voiceSelect.value || DEFAULT_VOICE;
  updateVoiceInfo(voice);

  // Show loading state
  window.generateBtn.classList.add('btn-loading');
  window.generateBtn.disabled = true;
  updateMode('Generating…');
  updateProgress(0);

  try {
    // Call generate directly instead of streaming loop
    const audioResult = await window.tts.generate(text, {
      voice,
      speed: 1,
    });

    // audioResult.audio is already a Float32Array
    const float32Arr = new Float32Array(audioResult.audio);

    // Convert to WAV Blob
    const wavBlob = float32ToWavPCM16(float32Arr, SAMPLE_RATE);
    const wavUrl = URL.createObjectURL(wavBlob);

    // Populate audio player
    window.audioPlayer.src = wavUrl;
    window.audioWrapper.style.display = 'block';

    // Enable download button
    window.downloadBtn.href = wavUrl;
    window.downloadBtn.download = `${voice}-kokoro-tts.wav`;
    window.downloadBtn.disabled = false;

    // Show status & play automatically
    updateMode('Synthesis complete!');
    updateProgress(100);
    window.audioPlayer.play();
  } catch (err) {
    console.error('TTS generation error:', err);
    alert('Failed to generate speech. Check the console for details.');
    updateMode('Error – see console');
  } finally {
    // Reset button
    window.generateBtn.classList.remove('btn-loading');
    window.generateBtn.disabled = false;
  }
}

function onClear() {
  window.textInput.value = '';
  window.charCount.textContent = '0 / 55 characters';
  window.audioWrapper.style.display = 'none';
  window.audioPlayer.src = '';
  URL.revokeObjectURL(window.audioPlayer.src);
  window.downloadBtn.href = '';
  window.downloadBtn.download = '';
  window.downloadBtn.disabled = true;
  updateMode('Ready');
  updateProgress(0);
}

/* -------------------------------------------------------------------------
   Character counter
   ------------------------------------------------------------------------- */
function updateCharCounter() {
  const text = window.textInput.value || '';
  const limit = 55;
  const count = text.length;
  window.charCount.textContent = `${count} / ${limit} characters`;
}

/* -------------------------------------------------------------------------
   Bootstrap: detect device, init Kokoro, wire UI
   ------------------------------------------------------------------------- */
async function bootstrap() {
  // 1. Detect device (WebGPU vs WASM)
  const deviceInfo = await detectDevice();

  // 2. Initialize Kokoro with the chosen mode/precision
  await initKokoro(deviceInfo);

  // 3. Wire up UI controls
  if (window.generateBtn) {
    window.generateBtn.addEventListener('click', onGenerate);
  }
  if (window.clearBtn) {
    window.clearBtn.addEventListener('click', onClear);
  }
  if (window.voiceSelect) {
    window.voiceSelect.addEventListener('change', () => {
      updateVoiceInfo(window.voiceSelect.value);
    });
  }
  if (window.textInput) {
    window.textInput.addEventListener('input', updateCharCounter);
    // Initial counter update
    updateCharCounter();
  }
}

// Start the app as soon as the DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}