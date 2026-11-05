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
const CHUNK_MAX_LENGTH = 50; // Max characters per phrase segment before chunking

// Break a chunk BEFORE these words when the buffer is at least half full:
// conjunctions that mark where one clause ends and the next begins.
const BREAK_WORDS = new Set([
  'and', 'but', 'or', 'because', 'so', 'yet', 'nor',
  'although', 'though', 'while', 'whereas', 'since', 'unless',
  'however', 'therefore', 'moreover', 'meanwhile', 'besides', 'otherwise',
  'if', 'when', 'whenever', 'where', 'whether', 'until',
  'which', 'whom', 'whose',
]);
// Subset that "Fix Punctuation" may insert a comma in front of (conservative:
// coordinators + because, plus relative pronouns that take commas in English).
const COMMA_CONJUNCTIONS = new Set([
  'and', 'but', 'or', 'because', 'so', 'yet', 'nor', 'which', 'whom', 'whose',
]);
const BREAK_WORD_MIN_BUFFER = Math.floor(CHUNK_MAX_LENGTH * 0.5); // 25 chars

// Exact inter‑chunk silence, inserted as sample counts during the Web Audio
// mix‑down (see the audio engine at the bottom of the helpers section).
const GAP_SENTENCE_MS = 260; // after . ! ?
const GAP_CLAUSE_MS = 100;   // after , ; : — –
const GAP_CONJUNCT_MS = 50;  // after conjunction splits (and/but/or/because…)

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
   Helper: split long text into phrase‑length segments (≤ CHUNK_MAX_LENGTH)

   Break priority (most natural first — the character limit is the FALLBACK):
     1. Sentence punctuation  . ! ?         → break after + GAP_SENTENCE_MS
     2. Clause punctuation    , ; : — –     → break after + GAP_CLAUSE_MS
     3. BREAK_WORDS starting a new clause   → break BEFORE the conjunction
        (once the buffer is ≥ BREAK_WORD_MIN_BUFFER chars) + GAP_CONJUNCT_MS
     4. Character limit                     → backtrack to the last natural
        clause joint inside the buffer (falls back to word boundary)
     5. Single token longer than maxLength  → hard split (URLs etc.)

   Each chunk carries the gapMs value of the pause that follows it; the audio
   engine inserts the silence as an exact number of samples, so gaps are
   deterministic and nothing else pads the audio between chunks.
   ------------------------------------------------------------------------- */
function splitTextIntoChunks(text, maxLength = CHUNK_MAX_LENGTH) {
  const chunks = [];
  if (!text) return chunks;

  const isSentenceEnd = (ch) => '.!?'.includes(ch);
  const isClausePause = (ch) => ',;:—–'.includes(ch);
  const pureWord = (w) => w.toLowerCase().replace(/[^a-z']/g, '');
  const isBreakWord = (tok) =>
    tok.length === pureWord(tok).length && BREAK_WORDS.has(pureWord(tok));

  let buffer = '';
  const flush = (gapMs = 0) => {
    const chunk = buffer.trim();
    if (chunk) chunks.push({ text: chunk, gapMs });
    buffer = '';
  };

  // Overflow handler: token doesn't fit in the buffer. If the buffer already
  // contains a clause joint (a BREAK_WORDS word), split there so the first
  // chunk ends at a natural pause — snapped only to `joint`, else word start.
  const overflowSplit = (token) => {
    const parts = buffer.split(' ');
    let joint = -1;
    for (let k = parts.length - 1; k >= 1; k--) {
      const bare = pureWord(parts[k]);
      if (parts[k].length === bare.length && BREAK_WORDS.has(bare)) {
        joint = k;
        break;
      }
    }
    if (joint > 0) {
      const right = parts.slice(joint).join(' ');
      buffer = parts.slice(0, joint).join(' ');
      flush(GAP_CONJUNCT_MS);
      buffer = `${right} ${token}`;
    } else {
      flush(0);
      buffer = token;
    }
  };

  const hardSplitLongWord = (word) => {
    // Emergency path: a single token longer than maxLength (e.g. a URL).
    for (let start = 0; start < word.length; start += maxLength) {
      flush(0);
      const piece = word.slice(start, start + maxLength);
      if (piece.length >= maxLength) {
        buffer = piece;
        flush(0);
      } else {
        buffer = piece;
      }
    }
  };

  for (const token of text.match(/\S+|\s+/g) || []) {
    if (/^\s+$/.test(token)) continue; // whitespace is re‑added via join

    if (token.length > maxLength) {
      hardSplitLongWord(token);
      continue;
    }

    const lastCh = token.at(-1);

    // 1) & 2) Natural pause punctuation: always end the chunk here — this
    //    break costs nothing, so it happens long before any limit pressure.
    if (isSentenceEnd(lastCh) || isClausePause(lastCh)) {
      const gapMs = isSentenceEnd(lastCh) ? GAP_SENTENCE_MS : GAP_CLAUSE_MS;
      const candidate = buffer ? `${buffer} ${token}` : token;
      if (candidate.length <= maxLength) {
        buffer = candidate;
        flush(gapMs);
      } else {
        overflowSplit(token);
        flush(gapMs);
      }
      continue;
    }

    // 3) Conjunction joint: break BEFORE "and / but / or / because…" as soon
    //    as the buffer is phrase‑sized, rather than mid‑clause at the limit.
    if (isBreakWord(token) && buffer.length >= BREAK_WORD_MIN_BUFFER) {
      flush(GAP_CONJUNCT_MS);
      buffer = token;
      continue;
    }

    // 4) Fallback: character limit — but still back up to the most recent
    //    clause joint inside the buffer before snapping at a word boundary.
    const candidate = buffer ? `${buffer} ${token}` : token;
    if (candidate.length <= maxLength) {
      buffer = candidate;
      continue;
    }
    overflowSplit(token);
  }
  flush(0);

  return chunks;
}

/* -------------------------------------------------------------------------
   Audio playback engine (Web Audio API)

   Every synthesized chunk goes through a full decode pipeline before any
   audio is handed to the player:

     Float32 PCM  →  trim edge silence  →  WAV blob  →  decodeAudioData
                  →  AudioBuffer (Web Audio)

   All chunk AudioBuffers are then concatenated into ONE unified AudioBuffer,
   inserting exact silence between chunks (GAP_SENTENCE_MS / GAP_CLAUSE_MS /
   GAP_CONJUNCT_MS converted to sample counts at the context sample rate).
   The combined buffer is written back out to a WAV for the browser player
   and the download link — one seamless file with deterministic pauses.
   ------------------------------------------------------------------------- */
const SILENCE_THRESHOLD = 0.004; // ~−48 dBFS noise floor; speech sits far above

let audioContext = null;
function getAudioContext() {
  if (!audioContext) {
    const AC = window.AudioContext || window.webkitAudioContext;
    // Run at the model's 24 kHz rate when the browser supports it: decode
    // and mix‑down then need no resampling and gaps map 1:1 to samples.
    try {
      audioContext = new AC({ sampleRate: SAMPLE_RATE });
    } catch (e) {
      audioContext = new AC(); // browser will resample to its own rate
    }
  }
  return audioContext;
}

function gapSampleCount(gapMs, sampleRate) {
  return Math.round((gapMs / 1000) * sampleRate);
}

// Trim near‑silent samples off both edges of a chunk so the silence that is
// inserted at chunk joints is truly exact (no Kokoro tail padding on top).
function trimEdgeSilence(samples) {
  let start = 0;
  let end = samples.length - 1;
  while (start <= end && Math.abs(samples[start]) < SILENCE_THRESHOLD) start++;
  while (end >= start && Math.abs(samples[end]) < SILENCE_THRESHOLD) end--;
  if (end < start) return samples; // whole chunk below threshold: leave as-is
  return samples.subarray(start, end + 1);
}

// chunk WAV blob → decoded AudioBuffer. Uses the promise form where the
// browser offers it, with decodeAudioData's callback form as a fallback.
function decodeWavBlob(wavBlob) {
  const ctx = getAudioContext();
  return wavBlob.arrayBuffer().then((bytes) => {
    return new Promise((resolve, reject) => {
      const decoded = ctx.decodeAudioData(bytes, resolve, reject);
      if (decoded && typeof decoded.then === 'function') {
        decoded.then(resolve, reject);
      }
    });
  });
}

// Concatenate chunk AudioBuffers into one unified AudioBuffer, inserting the
// exact gap samples between joints. Expected shape:
//   [{ buffer: AudioBuffer, gapMs: number }, ...]
function mixDownAudioBuffers(chunkResults) {
  const ctx = getAudioContext();
  const sampleRate = ctx.sampleRate;

  const audioTotal = chunkResults.reduce((sum, c) => sum + c.buffer.length, 0);
  // The final chunk gets no trailing gap, so only the non‑last chunks count.
  const gapTotal = chunkResults
    .slice(0, -1)
    .reduce((sum, c) => sum + gapSampleCount(c.gapMs, sampleRate), 0);

  const combined = ctx.createBuffer(1, audioTotal + gapTotal, sampleRate);
  const data = combined.getChannelData(0);

  let offset = 0;
  chunkResults.forEach((chunk, i) => {
    data.set(chunk.buffer.getChannelData(0), offset);
    offset += chunk.buffer.length;
    const last = i === chunkResults.length - 1;
    if (!last) offset += gapSampleCount(chunk.gapMs, sampleRate);
  });
  return combined;
}

/* -------------------------------------------------------------------------
   Text optimization ("Fix Punctuation")

   Cleans and re‑punctuates text client‑side so the chunker always finds
   natural clause boundaries. It NEVER adds, removes, reorders or replaces
   words — only punctuation, spacing and capitalization change.

   Rules:
     A. Collapse messy spacing; put punctuation tight to the word before it
        and one space after it.
     B. Collapse runs of identical punctuation ("fun!!!" → "fun!").
     C. Insert a comma before clause‑joining conjunctions (and, but, because,
        or, …) when the clause before them is a least 3 words long — short
        pairs like "tea and coffee" are left untouched.
     D. Upgrade a comma to a semicolon between two long independent clauses
        (second clause starts with a subject and contains a common verb).
     E. Capitalize sentence starts, give every sentence terminal punctuation,
        and fix missing spaces after ". " boundaries.
   ------------------------------------------------------------------------- */
const CONJ_MIN_WORDS = 4;

const SUBJECT_STARTERS = new Set([
  'i', 'we', 'you', 'he', 'she', 'it', 'they', 'this', 'that', 'these',
  'those', 'there', 'then', 'everything', 'nothing', 'something', 'anything',
  'everyone', 'someone', 'anyone', 'nobody', 'people', 'one',
]);

const COMMON_VERB_RE = new RegExp(
  '\\b(is|are|was|were|am|be|been|being|have|has|had|will|would|can|could|' +
  'should|may|might|must|do|does|did|' +
  'work(s|ed)?|run(s)?|start(s|ed)?|help(s|ed)?|keep(s)?|need(s|ed)?|' +
  'want(s|ed)?|think(s)?|know(s)?|like(s|d)?|sound(s|ed)?|stay(s|ed)?|' +
  'make(s)?|take(s)?|get(s)?|go(es)?|come(s)?|see(s)?|use(s|d)?|say(s)?|' +
  'feel(s)?|look(s|ed)?|live(s|d)?|seem(s|ed)?|happen(s|ed)?|upload(s|ed)?|' +
  'leave(s)?|hear(s|d)?|speak(s)?|talk(s|ed)?|read(s)?|write(s)?|play(s|ed)?)\\b',
  'i'
);

function wordCount(s) {
  return s.trim() ? s.trim().split(/\s+/).length : 0;
}

function insertPunctuationInside(text) {
  // Walks one clause and inserts missing punctuation BEFORE clause joints.
  // Returns sub-objects { text, delim } — every delimiter after the first is
  // newly inserted punctuation:
  //   ',' before conjunctions once the clause has ≥ CONJ_MIN_WORDS words
  //      (short pairs like "tea and coffee" stay untouched),
  //   ';' before a new subject that starts its own independent clause
  //      (run‑on rescue, e.g. "it is free you do not need a key").
  const tokens = text.split(/\s+/).filter(Boolean);
  if (tokens.length <= 1) return [{ text: tokens.join(' ') || text, delim: '' }];

  const groups = [];
  let current = [];
  let wordsSinceBreak = 0;

  const groupText = () => current.join(' ');
  const endsBreak = (tok) => /[,;:]$/.test(tok);
  const startsClauseJoint = (force) => {
    groups.push({ text: current.join(' '), delim: force });
    current = [];
    wordsSinceBreak = 0;
  };

  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    const bare = tok.toLowerCase().replace(/[^a-z']/g, '');
    const pureWord = bare.length === tok.length;
    const prev = i > 0 ? tokens[i - 1].toLowerCase().replace(/[^a-z']/g, '') : '';

    if (current.length > 0) {
      // clause joint at a conjunction word
      if (
        pureWord && COMMA_CONJUNCTIONS.has(bare) &&
        wordsSinceBreak >= CONJ_MIN_WORDS
      ) {
        startsClauseJoint(',');
      }
      // run-on rescue: a new subject taking over, clause already has
      // ≥ 5 words AND a verb, and it does not already sit behind a conjunction
      else if (
        pureWord && SUBJECT_STARTERS.has(bare) &&
        wordsSinceBreak >= 5 &&
        COMMON_VERB_RE.test(groupText()) &&
        !BREAK_WORDS.has(prev)
      ) {
        startsClauseJoint(';');
      }
    }

    current.push(tok);
    wordsSinceBreak++;
    if (endsBreak(tok)) wordsSinceBreak = 0; // existing break resets counting
  }
  if (current.length) groups.push({ text: current.join(' '), delim: '' });
  return groups.filter((g) => g.text);
}

function punctuateSentence(sentence) {
  // Split the sentence into clauses on existing , ; : (delimiters kept).
  const rawParts = sentence.split(/([,;:]+)/);
  const clauses = [];
  for (let i = 0; i < rawParts.length; i += 2) {
    const text = rawParts[i].trim();
    const delim = rawParts[i + 1] || '';
    if (text) clauses.push({ text, delim });
  }
  if (!clauses.length) return sentence;

  // C) clause‑joint punctuation inside each clause; the original delimiter
  //    of the clause stays attached to its final sub‑clause.
  const expanded = [];
  clauses.forEach((c) => {
    const subs = insertPunctuationInside(c.text);
    subs.forEach((sub, si) => {
      expanded.push({
        text: sub.text,
        delim: si === subs.length - 1 ? c.delim : sub.delim,
      });
    });
  });

  // D) long comma‑joined independent clauses get a semicolon.
  for (let i = 0; i < expanded.length - 1; i++) {
    const cur = expanded[i];
    const next = expanded[i + 1];
    if (cur.delim !== ',') continue;
    const first =
      (next.text.split(/\s+/)[0] || '').toLowerCase().replace(/[^a-z']/g, '');
    if (
      SUBJECT_STARTERS.has(first) &&
      COMMON_VERB_RE.test(next.text) &&
      wordCount(cur.text) >= CONJ_MIN_WORDS &&
      wordCount(next.text) >= 2
    ) {
      cur.delim = ';';
    }
  }

  let out = expanded.map((c) => (c.text + c.delim)).join(' ').trim();

  // E) capitalize the sentence start.
  const firstLetterIdx = out.search(/[A-Za-z]/);
  if (firstLetterIdx >= 0) {
    out =
      out.slice(0, firstLetterIdx) +
      out[firstLetterIdx].toUpperCase() +
      out.slice(firstLetterIdx + 1);
  }

  // E) terminal punctuation: a trailing , ; : or nothing becomes a period.
  out = out.replace(/[,;:\s]+$/, '');
  if (!/[.!?…]$/.test(out)) out += '.';
  return out;
}

function optimizeTextPunctuation(text) {
  if (!text || !text.trim()) return text;

  // A) collapse whitespace; punctuation tight to the preceding word.
  let t = text.replace(/\s+/g, ' ').trim();
  t = t.replace(/\s+([,.!?;:…])/g, '$1');
  t = t.replace(/([,;…:])(?=[A-Za-z0-9])/g, '$1 ');
  t = t.replace(/([a-z][.!?])(?=[A-Za-z0-9])/g, '$1 '); // "end.Next" → "end. Next"

  // B) collapse punctuation runs ("!!!" → "!").
  t = t.replace(/([,.!?;:…])\1+/g, '$1');

  // C/D/E) sentence by sentence.
  t = t
    .split(/(?<=[.!?…])\s+/)
    .filter(Boolean)
    .map(punctuateSentence)
    .join(' ');

  return t.trim();
}

function onFixPunctuation() {
  const raw = window.textInput.value;
  if (!raw.trim()) {
    alert('Nothing to optimize — enter some text first.');
    return;
  }
  const optimized = optimizeTextPunctuation(raw);
  window.textInput.value = optimized;
  updateCharCounter();
  updateMode(optimized !== raw ? 'Text optimized ✓' : 'Text already clean ✓');
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
  window.fixPunctBtn = document.getElementById('fixPunctBtn');

  // Update status to show initialization
  updateMode('Initializing Kokoro…');
  window.statusBar.classList.add('loading'); // shows the progress chip during download

  // Load the model from Hugging Face (onnx‑community Kokoro‑82M)
  // The from_pretrained method will download & cache the model.
  // We pass progress_callback to update UI.
  const progress = { loaded: 0, total: 100 };

  const tts = await KokoroTTS.from_pretrained(
    'onnx-community/Kokoro-82M-v1.0-ONNX',
    {
      dtype,
      device,
      // kokoro-js delivers a ProgressInfo object here:
      // { status, name, file, progress (0-100), loaded, total }
      // — not a plain percentage number.
      progress_callback: (p) => {
        const pct =
          typeof p === 'number'
            ? Math.round(p)
            : typeof p?.progress === 'number'
              ? Math.round(p.progress)
              : p && p.total
                ? Math.round((p.loaded / p.total) * 100)
                : 0;
        progress.loaded = pct;
        updateProgress(pct);
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
  window.statusBar.classList.remove('loading');

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
    // Split long text into phrase‑length segments so each call stays within
    // the model's comfortable input range — this prevents degraded/clipped
    // synthesis on long inputs.
    const chunks = splitTextIntoChunks(text);
    console.info(`Split text into ${chunks.length} chunk(s):`, chunks.map((c) => c.text));

    // Unlock Web Audio inside the click's user gesture (autoplay policies).
    const audioContext = getAudioContext();
    if (audioContext.state === 'suspended') {
      await audioContext.resume().catch(() => {});
    }

    // Synthesize chunks sequentially (never concurrently). Each chunk runs
    // the full decode pipeline: Float32 PCM → trim edge silence → WAV blob
    // → decodeAudioData → AudioBuffer.
    const chunkResults = [];
    for (let i = 0; i < chunks.length; i++) {
      updateMode(`Generating chunk ${i + 1} of ${chunks.length}…`);
      updateProgress(Math.round((i / chunks.length) * 100));

      const audioResult = await window.tts.generate(chunks[i].text, {
        voice,
        speed: 1,
      });

      // Trim Kokoro's own edge padding so joint silence stays exact, then
      // encode to a WAV blob and pre-decode it into an AudioBuffer.
      const samples = trimEdgeSilence(new Float32Array(audioResult.audio));
      const chunkWav = float32ToWavPCM16(samples, SAMPLE_RATE);
      const audioBuffer = await decodeWavBlob(chunkWav);

      // gapMs = the pause this chunk is followed by (exact sample count is
      // applied during mix-down, not baked into the chunk audio).
      chunkResults.push({ buffer: audioBuffer, gapMs: chunks[i].gapMs });
    }

    // Exactly one unified AudioBuffer: all chunk audio in sequence with the
    // precise silence (260/100/50 ms) between joints — no stray gaps.
    const combined = mixDownAudioBuffers(chunkResults);
    window.lastMix = { chunkResults, combined }; // handy for debugging

    // Final combined audio → single WAV blob for player + download.
    const sampleRate = combined.sampleRate;
    const float32Arr = combined.getChannelData(0).slice();

    const wavBlob = float32ToWavPCM16(float32Arr, sampleRate);
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
  const chunkCount = splitTextIntoChunks(text.trim()).length;
  window.charCount.textContent =
    `${text.length} characters` + (chunkCount > 1 ? ` • ${chunkCount} chunks` : '');
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
  if (window.fixPunctBtn) {
    window.fixPunctBtn.addEventListener('click', onFixPunctuation);
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