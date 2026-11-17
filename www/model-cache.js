/**
 * model-cache.js — IndexedDB-backed storage for large AI model files plus a
 * resumable, network-resilient chunked downloader.
 *
 * Design goals:
 *   • Whole-file caching: a download is only marked complete once every byte
 *     has arrived, so a cached file can never be a truncated/corrupt model.
 *   • Progress reporting via an onProgress(percent) callback while chunks
 *     stream in, so the UI can show real download percentages instead of 0%.
 *   • Offline resilience: cached files load with zero network access, and
 *     network failures are retried with backoff instead of crashing the app.
 *
 * IndexedDB database "PebbleTTS_Cache", object store "model_files".
 * Each record: { key, blob, url, etag, size, contentType, cachedAt }.
 */

const DB_NAME = 'PebbleTTS_Cache';
const DB_VERSION = 1;
const STORE_NAME = 'model_files';

/** Tracks live object URLs created per cache key, mapped by URL. */
const OBJECT_URL_STATES = new Map();

/* -------------------------------------------------------------------------
   IndexedDB plumbing
   ------------------------------------------------------------------------- */

let dbPromise = null;

/**
 * Opens (and caches) a connection to the cache database. The promise is
 * reused for the module lifetime so every operation shares one connection.
 * @returns {Promise<IDBDatabase>}
 */
function openCacheDB() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof window === 'undefined' || !window.indexedDB) {
      reject(new Error('IndexedDB is not available in this environment.'));
      return;
    }
    const open = window.indexedDB.open(DB_NAME, DB_VERSION);
    open.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    open.onsuccess = (event) => {
      // If the connection is ever closed out from under us (e.g. browser
      // evicted it), drop the cached promise so the next call reopens.
      const db = event.target.result;
      db.onclose = () => { dbPromise = null; };
      db.onversionchange = () => { db.close(); dbPromise = null; };
      resolve(db);
    };
    open.onerror = (event) => reject(event.target.error);
  });
  return dbPromise;
}

/**
 * Runs a callback with a transaction store and resolves on transaction
 * completion, rejecting if the transaction or request fails.
 * @template T
 * @param {IDBTransactionMode} mode
 * @param {(store: IDBObjectStore) => IDBRequest} fn
 * @returns {Promise<T>}
 */
function withStore(mode, fn) {
  return openCacheDB().then(
    (db) =>
      new Promise((resolve, reject) => {
        let request;
        let tx;
        try {
          tx = db.transaction(STORE_NAME, mode);
          request = fn(tx.objectStore(STORE_NAME));
        } catch (err) {
          reject(err);
          return;
        }
        tx.oncomplete = () => resolve(request?.result);
        tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted.'));
        tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed.'));
      })
  );
}

/**
 * Reads the full record ({ blob, etag, ... }) for a cache key, or null.
 * Never throws — IndexedDB failures behave as a cache miss.
 * @param {string} key
 * @returns {Promise<{blob: Blob, [k: string]: any}|null>}
 */
async function readRecordSafe(key) {
  try {
    // withStore resolves with the request's result on transaction completion.
    const record = await withStore('readonly', (store) => store.get(key));
    return record ?? null;
  } catch (err) {
    console.warn('[PebbleTTS] IndexedDB read failed (treating as cache miss):', err);
    return null;
  }
}

/**
 * Writes a record for a cache key. Never throws; returns whether it persisted.
 * @param {string} key
 * @param {{blob: Blob, [k: string]: any}} record
 * @returns {Promise<boolean>}
 */
async function writeRecordSafe(key, record) {
  try {
    await withStore('readwrite', (store) => store.put(record, key));
    return true;
  } catch (err) {
    // Quota exceeded / private mode etc. — the download can still be used
    // from memory; we just won't have it next session.
    console.warn('[PebbleTTS] IndexedDB write failed (continuing without cache):', err);
    return false;
  }
}

/* -------------------------------------------------------------------------
   Public cache API
   ------------------------------------------------------------------------- */

/**
 * Returns the fully cached Blob for a key, or null if not present.
 * @param {string} key
 * @returns {Promise<Blob|null>}
 */
export async function getCachedFile(key) {
  const record = await readRecordSafe(key);
  return record?.blob && record.blob.size > 0 ? record.blob : null;
}

/**
 * Stores a Blob under a cache key (atomic put on transaction completion).
 * @param {string} key
 * @param {Blob} blob
 * @param {{url?: string, etag?: string, contentType?: string, [k: string]: any}} [meta={}]
 * @returns {Promise<void>}
 */
export async function cacheFile(key, blob, meta = {}) {
  const record = {
    key,
    blob,
    size: blob.size,
    cachedAt: Date.now(),
    ...meta,
  };
  await withStore('readwrite', (store) => store.put(record, key));
}

/**
 * Removes a cached file (and any object URL handed out for it).
 * @param {string} key
 * @returns {Promise<void>}
 */
export async function deleteCachedFile(key) {
  const record = await readRecordSafe(key);
  if (record?.objectUrl) revokeObjectUrl(record.objectUrl);
  await withStore('readwrite', (store) => store.delete(key)).catch(() => {});
}

/**
 * Returns the full stored record (Blob plus metadata such as `etag`) or null.
 * Useful for callers that need to echo cache metadata into a `Response`
 * (transformers.js custom cache adapters, for example).
 * @param {string} key
 * @returns {Promise<object|null>}
 */
export async function getCachedRecord(key) {
  return readRecordSafe(key);
}

/**
 * Creates (or reuses) an object URL for a cached blob and remembers the key it
 * belonged to, so `deleteCachedFile` can revoke stale URLs.
 * @param {string} key
 * @param {Blob} blob
 * @returns {string}
 */
export function createObjectUrlFor(key, blob) {
  const record = OBJECT_URL_STATES.get(key);
  if (record?.url) return record.url;
  const url = URL.createObjectURL(blob);
  OBJECT_URL_STATES.set(key, { url, blob });
  return url;
}

/** Revokes an object URL produced by {@link createObjectUrlFor}. */
export function revokeObjectUrl(url) {
  if (!url || !url.startsWith?.('blob:')) return;
  try { URL.revokeObjectURL(url); } catch { /* noop */ }
  for (const [key, state] of OBJECT_URL_STATES) {
    if (state.url === url) OBJECT_URL_STATES.delete(key);
  }
}

/* -------------------------------------------------------------------------
   Resumable chunked downloader
   ------------------------------------------------------------------------- */

/** Error thrown by {@link loadModelWithResume} when a download ultimately fails. */
export class ModelDownloadError extends Error {
  constructor(message, { cause, status, url } = {}) {
    super(message);
    this.name = 'ModelDownloadError';
    this.cause = cause;
    this.status = status;
    this.url = url;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isNonRetryableStatus(status) {
  // 4xx (except 429 rate limiting) means the request itself is invalid —
  // retrying immediately would only repeat the same failure.
  return status >= 400 && status < 500 && status !== 429;
}

/**
 * Streams `url` fully into a Blob, reporting percent progress.
 * Integrity-checked against Content-Length (or opts.expectedSize) before
 * returning, so partial/truncated responses are rejected and retried.
 *
 * @param {string} url
 * @param {(percent: number) => void} report
 * @param {{expectedSize?: number, maxRetries?: number, retryDelayMs?: number,
 *          signal?: AbortSignal, onRetry?: (info: {attempt: number,
 *          maxRetries: number, error: unknown}) => void}} [opts]
 * @returns {Promise<{blob: Blob, etag: string|null, bytes: number}>}
 */
async function downloadToBlob(url, report, {
  expectedSize = 0,
  maxRetries = 2,
  retryDelayMs = 900,
  signal = null,
  onRetry = null,
} = {}) {
  let lastError = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      // Linear backoff; transient dropouts and mobile flapping usually
      // recover within a second or two.
      await sleep(retryDelayMs * attempt);
      if (signal?.aborted) throw new ModelDownloadError('Download aborted.', { url });
    }

    try {
      const response = await fetch(url, signal ? { signal } : {});
      if (!response.ok) {
        throw new ModelDownloadError(
          `Download failed: HTTP ${response.status} ${response.statusText}`, {
            status: response.status,
            url,
          }
        );
      }

      // Fallback for environments without streaming responses (older WebView).
      if (!response.body || typeof response.body.getReader !== 'function') {
        const blob = await response.blob();
        report(100);
        return {
          blob,
          etag: response.headers.get('ETag'),
          bytes: blob.size,
        };
      }

      const headerLength = parseInt(response.headers.get('Content-Length'), 10);
      const total = Number.isFinite(headerLength) ? headerLength : expectedSize;

      /** @type {Uint8Array[]} */
      const chunks = [];
      let received = 0;
      let lastPercent = 0;

      const reader = response.body.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          if (total > 0) {
            const percent = Math.min(99, Math.floor((received / total) * 100));
            if (percent > lastPercent) {
              lastPercent = percent;
              report(percent);
            }
          }
        }
      } finally {
        // Release the underlying connection promptly.
        try { reader.cancel(); } catch { /* already closed */ }
      }

      // EOF integrity check: a Content-Length mismatch means we read a
      // truncated body (proxy cut-off, connection drop). Never cache this.
      if (total > 0 && received !== total) {
        throw new ModelDownloadError(
          `Download truncated: received ${received} of ${total} bytes.`, { url }
        );
      }

      const blob = new Blob(chunks, {
        type: response.headers.get('Content-Type') ?? 'application/octet-stream',
      });

      return {
        blob,
        etag: response.headers.get('ETag') ?? response.headers.get('X-Linked-ETag'),
        bytes: received,
      };
    } catch (err) {
      lastError = err;
      const nonRetryable =
        err instanceof ModelDownloadError && isNonRetryableStatus(err.status);
      if (signal?.aborted) {
        throw err instanceof ModelDownloadError
          ? err
          : new ModelDownloadError('Download aborted.', { cause: err, url });
      }
      if (nonRetryable || attempt === maxRetries) break;
      // Surface the retry so the UI can replace a stale percentage with a
      // "connection dropped, retrying…" message. Observer errors are ignored.
      if (typeof onRetry === 'function') {
        try {
          onRetry({ attempt: attempt + 1, maxRetries: maxRetries + 1, error: err });
        } catch { /* noop */ }
      }
      console.warn(
        `[PebbleTTS] Download attempt ${attempt + 1}/${maxRetries + 1} failed, retrying:`,
        err
      );
    }
  }

  throw lastError instanceof ModelDownloadError
    ? lastError
    : new ModelDownloadError('Download failed after retries.', { cause: lastError, url });
}

/* -------------------------------------------------------------------------
   Public downloader
   ------------------------------------------------------------------------- */

/**
 * Loads a model/voice file with resumable, cached semantics:
 *
 *   1. If the file is already fully cached in IndexedDB — report 100% and
 *      return a blob: object URL immediately (zero network).
 *   2. Otherwise fetch over the network, streaming chunks through
 *      response.body.getReader(), report percentage progress via
 *      `onProgress(percent)`, assemble the chunks into a Blob, persist it
 *      into IndexedDB, and return its object URL.
 *
 * Transient network failures are retried with backoff; all failures are
 * rejected as ModelDownloadError so callers can render a message instead of
 * crashing.
 *
 * @param {string} url           Remote file URL (e.g. the ONNX model on HF).
 * @param {string} cacheKey      Stable key under "model_files" for this file.
 * @param {(percent: number) => void} [onProgress]
 * @param {{expectedSize?: number, maxRetries?: number, retryDelayMs?: number,
 *          signal?: AbortSignal, onRetry?: (info: {attempt: number,
 *          maxRetries: number, error: unknown}) => void}} [opts]
 *          `onRetry` fires once per retry attempt (before the backoff sleep)
 *          so callers can update status text when the connection drops.
 * @returns {Promise<string>} blob: object URL usable by fetch/ORT/etc.
 */
export async function loadModelWithResume(url, cacheKey, onProgress, opts = {}) {
  const report = typeof onProgress === 'function' ? onProgress : () => {};
  if (typeof url !== 'string' || !url) {
    throw new TypeError('loadModelWithResume: "url" must be a non-empty string.');
  }
  if (typeof cacheKey !== 'string' || !cacheKey) {
    throw new TypeError('loadModelWithResume: "cacheKey" must be a non-empty string.');
  }

  // 1) Fully cached → serve locally, no network.
  const cached = await readRecordSafe(cacheKey);
  if (cached?.blob && cached.blob.size > 0) {
    const objectUrl = createObjectUrlFor(cacheKey, cached.blob);
    report(100);
    return objectUrl;
  }

  // 2) Network download with progress.
  const { blob, etag } = await downloadToBlob(url, report, opts);

  // 3) Persist (a failed write must not fail the load — the blob is live).
  await writeRecordSafe(cacheKey, {
    key: cacheKey,
    blob,
    url,
    etag: etag ?? undefined,
    size: blob.size,
    contentType: blob.type,
    cachedAt: Date.now(),
  });

  // 4) Hand back a local object URL.
  const objectUrl = createObjectUrlFor(cacheKey, blob);
  report(100);
  return objectUrl;
}

/**
 * Cheap probe — true when the cache database can be opened. Never throws.
 * @returns {Promise<boolean>}
 */
export async function isModelCacheAvailable() {
  try {
    await openCacheDB();
    return true;
  } catch {
    return false;
  }
}
