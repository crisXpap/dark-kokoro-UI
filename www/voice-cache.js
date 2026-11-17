/**
 * voice-cache.js — "Download all voices?" prompt card.
 *
 * Kokoro ships one small tensor file per voice (~0.5 MB each, ~27 MB for the
 * full set). kokoro-js fetches them lazily from Hugging Face and keeps them in
 * its CacheStorage bucket "kokoro-voices"; this module offers to fetch the
 * whole set up front so every voice works offline on the first try.
 *
 * Safety properties:
 *   • Every voice is downloaded fully into memory and validated
 *     (`response.ok`, non-empty body, matching Content-Length) before anything
 *     is written. A dropped connection or a closed tab can therefore never
 *     leave a truncated, corrupt voice entry behind.
 *   • Writes go to CacheStorage first (that is the bucket kokoro-js reads) and
 *     are mirrored into the IndexedDB "voice:<name>" records used by the
 *     app's lazy prefetch path.
 *   • The card collapses itself once every voice is present, including on a
 *     later app load where the cache is already complete.
 */

import { getCachedRecord, cacheFile } from './model-cache.js';

const VOICE_CACHE_NAME = 'kokoro-voices';
const DEFAULT_URL_BASE =
  'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/';
const APPROX_VOICE_BYTES = 524288; // ~0.5 MB per Kokoro voice tensor
const STORAGE_HEADROOM = 1.1; // keep 10% of the quota free
const DOWNLOAD_ATTEMPTS = 2;
const RETRY_DELAY_MS = 700;

let voiceNames = [];
let urlBase = DEFAULT_URL_BASE;
let downloading = false;
let promptBound = false;
let wakeLock = null;
let wakeLockListener = null;

const $ = (id) => document.getElementById(id);

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function voiceUrl(name) {
  return `${urlBase}${name}.bin`;
}

function formatMegabytes(bytes) {
  return Math.max(1, Math.round(bytes / (1024 * 1024)));
}

/* -------------------------------------------------------------------------
   Card states
   ------------------------------------------------------------------------- */

/** Shows the prompt with its "Yes" button and an idle subtitle. */
function showPrompt(remaining) {
  const card = $('voiceCacheCard');
  const yes = $('voiceCacheYes');
  const status = $('voiceCacheStatus');
  const sub = $('voiceCacheSub');
  if (!card) return;

  card.hidden = false;
  card.classList.remove('is-collapsing');
  card.removeAttribute('aria-busy');

  if (yes) {
    yes.hidden = false;
    yes.disabled = false;
    yes.textContent = 'Yes';
  }
  if (status) status.hidden = true;
  if (sub) {
    sub.textContent = `~${formatMegabytes(remaining * APPROX_VOICE_BYTES)} MB • Available offline`;
  }
}

/** Replaces the "Yes" button with a spinner + "Downloading N / total…". */
function showProgress(done, total) {
  const card = $('voiceCacheCard');
  const yes = $('voiceCacheYes');
  const status = $('voiceCacheStatus');
  const text = $('voiceCacheStatusText');

  if (yes) yes.hidden = true;
  if (status) status.hidden = false;
  if (text) text.textContent = `Downloading ${done} / ${total}…`;
  if (card) card.setAttribute('aria-busy', 'true');
}

/**
 * Smoothly collapses and then removes the card from layout. `immediate`
 * skips the animation (used when the cache is already complete at startup).
 */
function collapseCard(immediate = false) {
  const card = $('voiceCacheCard');
  if (!card || card.hidden) return;

  if (immediate || card.classList.contains('is-collapsing')) {
    if (immediate) {
      card.hidden = true;
      card.classList.remove('is-collapsing');
    }
    return;
  }

  card.classList.add('is-collapsing');

  let finished = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    clearTimeout(timer);
    card.removeEventListener('transitionend', onTransitionEnd);
    card.hidden = true;
    card.classList.remove('is-collapsing');
    card.removeAttribute('aria-busy');
  };
  const onTransitionEnd = (event) => {
    if (event.target === card) finish();
  };

  card.addEventListener('transitionend', onTransitionEnd);
  // Failsafe for prefers-reduced-motion (no transition events fire).
  const timer = setTimeout(finish, 450);
}

/* -------------------------------------------------------------------------
   Cache inspection / storage
   ------------------------------------------------------------------------- */

async function openVoiceCache() {
  if (typeof caches === 'undefined') return null;
  try {
    return await caches.open(VOICE_CACHE_NAME);
  } catch (err) {
    console.warn('[PebbleTTS] CacheStorage unavailable:', err);
    return null;
  }
}

/**
 * True when a complete voice tensor is already cached. CacheStorage is the
 * source of truth for kokoro-js; the IndexedDB record is used as a fallback
 * and, when found, copied into CacheStorage so the loader can use it offline.
 */
async function isVoiceCached(cache, name) {
  const url = voiceUrl(name);

  if (cache) {
    try {
      if (await cache.match(url)) return true;
    } catch (err) {
      console.warn(`[PebbleTTS] Voice cache lookup failed for ${name}:`, err);
    }
  }

  const record = await getCachedRecord(`voice:${name}`);
  if (record?.blob && record.blob.size > 0) {
    if (cache) {
      try {
        await cache.put(url, new Response(record.blob, {
          headers: { 'Content-Type': 'application/octet-stream' },
        }));
      } catch (err) {
        console.warn(`[PebbleTTS] Could not seed CacheStorage for ${name}:`, err);
      }
    }
    return true;
  }

  return false;
}

/** Returns the subset of voices that are not fully cached yet. */
async function listUncached() {
  const cache = await openVoiceCache();
  const checks = await Promise.all(
    voiceNames.map(async (name) => ((await isVoiceCached(cache, name)) ? null : name))
  );
  return checks.filter(Boolean);
}

/**
 * Pre-flight quota check. Returns true when the API is unavailable so an
 * unsupported browser is never blocked from downloading.
 */
async function hasSpaceFor(bytes) {
  try {
    const estimate = await navigator.storage?.estimate?.();
    const { quota, usage } = estimate || {};
    if (!Number.isFinite(quota) || !Number.isFinite(usage)) return true;
    return quota - usage >= bytes * STORAGE_HEADROOM;
  } catch (err) {
    console.info('[PebbleTTS] Storage estimate unavailable:', err);
    return true;
  }
}

/* -------------------------------------------------------------------------
   Screen wake lock (best-effort; ignored where unsupported)
   ------------------------------------------------------------------------- */

async function acquireWakeLock() {
  try {
    if (!navigator.wakeLock || typeof navigator.wakeLock.request !== 'function') return;

    wakeLock = await navigator.wakeLock.request('screen');
    wakeLock.addEventListener?.('release', () => { wakeLock = null; });

    // Browsers release the lock when the tab is hidden; re-acquire it if the
    // user comes back while the download is still running.
    wakeLockListener = async () => {
      if (!downloading || document.visibilityState !== 'visible' || wakeLock) return;
      try {
        wakeLock = await navigator.wakeLock.request('screen');
        wakeLock.addEventListener?.('release', () => { wakeLock = null; });
      } catch { /* permission or low battery — not fatal */ }
    };
    document.addEventListener('visibilitychange', wakeLockListener);
  } catch (err) {
    console.info('[PebbleTTS] Screen wake lock unavailable:', err);
    wakeLock = null;
  }
}

async function releaseWakeLock() {
  if (wakeLockListener) {
    document.removeEventListener('visibilitychange', wakeLockListener);
    wakeLockListener = null;
  }
  if (wakeLock) {
    try {
      await wakeLock.release();
    } catch { /* already released */ }
    wakeLock = null;
  }
}

/* -------------------------------------------------------------------------
   Atomic per-voice download
   ------------------------------------------------------------------------- */

/**
 * Downloads one voice into an ArrayBuffer, validates it, and only then commits
 * it to CacheStorage and IndexedDB. Retries transient failures.
 */
async function downloadVoice(name, cache) {
  const url = voiceUrl(name);
  let lastError = null;

  for (let attempt = 0; attempt < DOWNLOAD_ATTEMPTS; attempt++) {
    if (attempt > 0) await sleep(RETRY_DELAY_MS * attempt);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      // Full body in memory — never a partially written stream.
      const buffer = await response.arrayBuffer();
      if (!buffer || buffer.byteLength === 0) {
        throw new Error('Empty voice file');
      }

      const declared = Number(response.headers.get('Content-Length'));
      if (Number.isFinite(declared) && declared > 0 && declared !== buffer.byteLength) {
        throw new Error(`Truncated voice file (${buffer.byteLength}/${declared} bytes)`);
      }

      const blob = new Blob([buffer], { type: 'application/octet-stream' });
      let stored = false;

      // CacheStorage is what kokoro-js's internal voice loader reads.
      if (cache) {
        await cache.put(url, new Response(blob, {
          headers: {
            'Content-Type': 'application/octet-stream',
            'Content-Length': String(buffer.byteLength),
          },
        }));
        stored = true;
      }

      // Mirror the record used by prefetchVoice(); best-effort.
      try {
        await cacheFile(`voice:${name}`, blob, {
          url,
          size: buffer.byteLength,
          contentType: 'application/octet-stream',
        });
        stored = true;
      } catch (err) {
        console.warn(`[PebbleTTS] IndexedDB mirror failed for ${name}:`, err);
      }

      if (!stored) throw new Error('No cache storage available');
      return buffer.byteLength;
    } catch (err) {
      lastError = err;
      console.warn(
        `[PebbleTTS] Voice download failed (${name}, attempt ${attempt + 1}/${DOWNLOAD_ATTEMPTS}):`,
        err
      );
    }
  }

  throw lastError || new Error(`Could not download ${name}`);
}

/* -------------------------------------------------------------------------
   Main flow
   ------------------------------------------------------------------------- */

async function startDownload() {
  if (downloading) return;

  const yes = $('voiceCacheYes');
  const sub = $('voiceCacheSub');
  if (!yes) return;

  downloading = true;
  yes.disabled = true;

  try {
    showProgress(0, voiceNames.length);
    if (sub) sub.textContent = 'Checking available space…';

    const uncached = await listUncached();
    const total = uncached.length;

    if (!total) {
      collapseCard();
      return;
    }

    // Count both copies (CacheStorage + IndexedDB mirror) so the pre-flight
    // check matches what will actually be written.
    const neededBytes = total * APPROX_VOICE_BYTES * 2;
    if (!(await hasSpaceFor(neededBytes))) {
      showPrompt(total);
      if (sub) sub.textContent = 'Not enough storage — free some space and try again';
      return;
    }

    showProgress(0, total);
    if (sub) sub.textContent = 'Keep this screen open';
    await acquireWakeLock();

    const cache = await openVoiceCache();
    let completed = 0;
    let failures = 0;

    for (const name of uncached) {
      try {
        // The lazy prefetcher may have cached it while we were working.
        if (!(await isVoiceCached(cache, name))) {
          await downloadVoice(name, cache);
        }
        completed++;
      } catch (err) {
        failures++;
        console.warn(`[PebbleTTS] Could not cache voice ${name}:`, err);
      }
      showProgress(completed + failures, total);
    }

    if (failures === 0) {
      collapseCard();
    } else {
      showPrompt(failures);
      if (sub) {
        sub.textContent =
          `${failures} voice${failures === 1 ? '' : 's'} failed • Tap Yes to retry`;
      }
      if (yes) yes.textContent = 'Retry';
    }
  } catch (err) {
    console.error('[PebbleTTS] Voice cache download failed:', err);
    showPrompt(voiceNames.length);
    if (sub) sub.textContent = 'Download failed — tap Yes to retry';
    if (yes) yes.textContent = 'Retry';
  } finally {
    downloading = false;
    await releaseWakeLock();
    if (yes) yes.disabled = false;
  }
}

/**
 * Initializes the prompt for the voice list known after model load. Safe to
 * call again (e.g. after a retry of model initialization).
 *
 * @param {string[]} names  All Kokoro voice names (e.g. ["af_heart", …]).
 * @param {{urlBase?: string}} [options]
 * @returns {Promise<void>}
 */
export async function initVoiceCachePrompt(names, options = {}) {
  if (!Array.isArray(names) || names.length === 0) return;

  voiceNames = Array.from(new Set(names.filter((name) => typeof name === 'string' && name)));
  urlBase = options.urlBase
    ? (options.urlBase.endsWith('/') ? options.urlBase : `${options.urlBase}/`)
    : DEFAULT_URL_BASE;

  if (!promptBound) {
    promptBound = true;
    $('voiceCacheYes')?.addEventListener('click', () => { startDownload(); });
  }

  if (downloading) return;

  try {
    const uncached = await listUncached();
    if (!uncached.length) {
      collapseCard(true);
      return;
    }
    showPrompt(uncached.length);
  } catch (err) {
    console.warn('[PebbleTTS] Voice cache status check failed:', err);
  }
}
