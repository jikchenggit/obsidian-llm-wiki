import { requestUrl } from 'obsidian';
import { unzipSync } from 'fflate';
import {
  MINERU_API_BASE_URL,
  MINERU_MAX_ARCHIVE_FILES,
  MINERU_MAX_PDF_BYTES,
  MINERU_MAX_ZIP_BYTES,
  MINERU_POLL_INTERVAL_MS,
  MINERU_TIMEOUT_MS,
  PDF_CACHE_MAX_SINGLE_ENTRY_BYTES,
} from '../constants';
import {
  createPdfCache,
  hashCacheKey,
  sha256Bytes,
} from './pdf-cache';
import { bytesToBase64 } from './pdf-converter';
import type { ConversionResult, PdfConversionContext } from './pdf-converter';

interface MineruEnvelope {
  code: number | string;
  msg?: string;
  data?: Record<string, unknown>;
}

/**
 * Machine-readable classification for known limit failures. Coded errors
 * are *expected* source rejections (file over a MinerU cap) and route
 * through the engine's skip pipeline with a localized Notice; uncoded
 * errors are unexpected failures and keep the throw semantics.
 */
export type MineruErrorCode = 'page-limit' | 'size-limit';

export class MineruPdfError extends Error {
  readonly code?: MineruErrorCode;
  constructor(message: string, code?: MineruErrorCode) {
    super(message);
    this.name = 'MineruPdfError';
    if (code !== undefined) this.code = code;
  }
}

// Simplification #7: phase → i18n-key lookup. Replaces the nested ternary
// that wiki-engine.ts used to inline, and gives exhaustiveness checks a
// hook if a future MinerU API state is added (TS will flag the missing
// case at the lookup site).
//
// The value is local; we export the type so callers can build their own
// lookups if they need a different i18n mapping later (e.g. a future
// HTML ingest that wants different progress strings).
export type MineruPhase = 'uploading' | 'waiting' | 'downloading';

export const MINERU_PHASE_KEY = {
  uploading: 'mineruUploadingInProgress',
  waiting: 'mineruWaitingInProgress',
  downloading: 'mineruDownloadingInProgress',
} as const satisfies Record<MineruPhase, string>;

export function extractMineruMarkdown(zipBytes: Uint8Array): string {
  let fileCount = 0;
  let markdownCount = 0;
  try {
    const files = unzipSync(zipBytes, {
      filter: file => {
        if (++fileCount > MINERU_MAX_ARCHIVE_FILES) {
          throw new MineruPdfError('MinerU result archive contains too many files.');
        }
        if (file.name.split('/').pop() !== 'full.md') return false;
        markdownCount++;
        if (markdownCount > 1) {
          throw new MineruPdfError('MinerU result must contain exactly one full.md document.');
        }
        if (file.originalSize > PDF_CACHE_MAX_SINGLE_ENTRY_BYTES) {
          throw new MineruPdfError('MinerU full.md exceeds the size limit.');
        }
        return true;
      },
    });
    if (markdownCount !== 1) {
      throw new MineruPdfError('MinerU result must contain exactly one full.md document.');
    }
    return new TextDecoder('utf-8', { fatal: true }).decode(Object.values(files)[0]);
  } catch (error) {
    if (error instanceof MineruPdfError) throw error;
    if (error instanceof TypeError) {
      throw new MineruPdfError('MinerU full.md is not valid UTF-8.');
    }
    throw new MineruPdfError('MinerU returned an invalid result archive.');
  }
}

function abortError(): DOMException {
  return new DOMException('MinerU conversion was cancelled.', 'AbortError');
}

function withDeadline<T>(promise: Promise<T>, deadline: number, signal?: AbortSignal): Promise<T> {
  throwIfAborted(signal);
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    let timeout = 0;
    const finish = (settle: () => void) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      signal?.removeEventListener('abort', onAbort);
      settle();
    };
    const onAbort = () => finish(() => reject(abortError()));
    timeout = window.setTimeout(
      () => finish(() => reject(new MineruPdfError('MinerU conversion timed out after 30 minutes.'))),
      Math.max(0, deadline - Date.now()),
    );
    signal?.addEventListener('abort', onAbort, { once: true });
    promise.then(
      value => finish(() => resolve(value)),
      error => finish(() => reject(error instanceof Error ? error : new MineruPdfError('MinerU request failed.'))),
    );
  });
}

export function resolveMineruBaseUrl(configured?: string): string {
  const trimmed = configured?.trim();
  if (!trimmed) return MINERU_API_BASE_URL;
  return trimmed.replace(/\/+$/, '').replace(/\/file_parse$/, '').replace(/\/+$/, '');
}

export function isSelfHostedMineruApi(baseUrl: string): boolean {
  const normalized = baseUrl.trim().replace(/\/+$/, '');
  if (normalized.endsWith('/v4') || normalized.includes('mineru.net')) {
    return false;
  }
  return true;
}

export function buildMultipartFormData(
  boundary: string,
  fields: Record<string, string>,
  file: { name: string; bytes: Uint8Array; fieldName?: string },
): Uint8Array {
  const encoder = new TextEncoder();
  const parts: Uint8Array[] = [];

  for (const [key, value] of Object.entries(fields)) {
    const fieldHeader = `--${boundary}\r\nContent-Disposition: form-data; name="${key}"\r\n\r\n${value}\r\n`;
    parts.push(encoder.encode(fieldHeader));
  }

  const fileFieldName = file.fieldName || 'files';
  const fileHeader = `--${boundary}\r\nContent-Disposition: form-data; name="${fileFieldName}"; filename="${file.name}"\r\nContent-Type: application/pdf\r\n\r\n`;
  parts.push(encoder.encode(fileHeader));
  parts.push(file.bytes);
  parts.push(encoder.encode(`\r\n--${boundary}--\r\n`));

  let totalLength = 0;
  for (const part of parts) {
    totalLength += part.byteLength;
  }

  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.byteLength;
  }

  return result;
}

export function extractMarkdownFromMineruApiResponse(json: unknown, filename?: string): string {
  if (!json || typeof json !== 'object') {
    throw new MineruPdfError('MinerU response did not contain markdown content.');
  }

  const obj = json as Record<string, unknown>;

  if (typeof obj.md_content === 'string' && obj.md_content.trim()) return obj.md_content;
  if (typeof obj.md === 'string' && obj.md.trim()) return obj.md;
  if (typeof obj.markdown === 'string' && obj.markdown.trim()) return obj.markdown;

  const results = obj.results;
  if (results && typeof results === 'object') {
    const resObj = results as Record<string, unknown>;
    if (filename && filename in resObj && typeof resObj[filename] === 'object' && resObj[filename] !== null) {
      const fileRes = resObj[filename] as Record<string, unknown>;
      const content = fileRes.md_content ?? fileRes.md ?? fileRes.markdown;
      if (typeof content === 'string' && content.trim()) return content;
    }

    if (filename) {
      const baseName = filename.replace(/\.[^/.]+$/, '');
      if (baseName in resObj && typeof resObj[baseName] === 'object' && resObj[baseName] !== null) {
        const fileRes = resObj[baseName] as Record<string, unknown>;
        const content = fileRes.md_content ?? fileRes.md ?? fileRes.markdown;
        if (typeof content === 'string' && content.trim()) return content;
      }
    }

    for (const val of Object.values(resObj)) {
      if (val && typeof val === 'object' && val !== null) {
        const sub = val as Record<string, unknown>;
        const content = sub.md_content ?? sub.md ?? sub.markdown;
        if (typeof content === 'string' && content.trim()) return content;
      }
    }
  }

  if (obj.data && typeof obj.data === 'object') {
    try {
      return extractMarkdownFromMineruApiResponse(obj.data, filename);
    } catch {
      // pass through to throw below
    }
  }

  throw new MineruPdfError('MinerU response did not contain markdown content.');
}

async function convertWithMineruV1(
  bytes: Uint8Array,
  filename: string,
  v1BaseUrl: string,
  token: string | undefined,
  deadline: number,
  signal?: AbortSignal,
  onPhase?: (phase: MineruPhase) => void,
): Promise<{ status: 'ok'; markdown: string } | { status: 'not-found' }> {
  throwIfAborted(signal);
  onPhase?.('uploading');

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const jobPayload = JSON.stringify({
    files: [
      {
        source: {
          type: 'inline',
          name: filename,
          data: bytesToBase64(bytes),
        },
      },
    ],
  });

  const jobResponse = await withDeadline(requestUrl({
    url: `${v1BaseUrl}/parse/jobs`,
    method: 'POST',
    headers,
    body: jobPayload,
    throw: false,
  }), deadline, signal);

  if (jobResponse.status === 404) {
    return { status: 'not-found' };
  }

  if (jobResponse.status < 200 || jobResponse.status >= 300) {
    const errData = jobResponse.json as { detail?: string; message?: string } | undefined;
    const msg = typeof errData?.detail === 'string'
      ? errData.detail
      : typeof errData?.message === 'string'
        ? errData.message
        : `MinerU request failed with HTTP ${jobResponse.status}.`;
    throw new MineruPdfError(msg);
  }

  const jobJson = jobResponse.json as {
    job_id?: string;
    status?: string;
    files?: Array<{
      status?: string;
      output_files?: { markdown?: { file_id?: string } };
      error?: { message?: string };
    }>;
  } | undefined;

  const jobId = stringValue(jobJson?.job_id);
  if (!jobId) {
    throw new MineruPdfError('MinerU returned an invalid job response.');
  }

  let fileId: string | undefined;
  if (jobJson?.status === 'completed') {
    const file = jobJson.files?.[0];
    fileId = stringValue(file?.output_files?.markdown?.file_id);
  }

  while (!fileId && Date.now() < deadline) {
    throwIfAborted(signal);
    onPhase?.('waiting');

    const pollResponse = await withDeadline(requestUrl({
      url: `${v1BaseUrl}/parse/jobs/${encodeURIComponent(jobId)}`,
      method: 'GET',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      throw: false,
    }), deadline, signal);

    if (pollResponse.status < 200 || pollResponse.status >= 300) {
      throw new MineruPdfError(`MinerU request failed with HTTP ${pollResponse.status}.`);
    }

    const pollJson = pollResponse.json as {
      status?: string;
      files?: Array<{
        status?: string;
        output_files?: { markdown?: { file_id?: string } };
        error?: { message?: string };
      }>;
    } | undefined;

    const taskStatus = pollJson?.status;
    const fileResult = pollJson?.files?.[0];

    if (taskStatus === 'completed' || fileResult?.status === 'completed') {
      fileId = stringValue(fileResult?.output_files?.markdown?.file_id);
      if (!fileId) {
        throw new MineruPdfError('MinerU returned no markdown file ID.');
      }
      break;
    }

    if (taskStatus === 'failed' || fileResult?.status === 'failed') {
      const errMsg = stringValue(fileResult?.error?.message);
      throw new MineruPdfError(errMsg ?? 'MinerU conversion failed.', classifyMineruFailure(errMsg));
    }

    if (taskStatus === 'canceled' || taskStatus === 'cancelled') {
      throw new MineruPdfError('MinerU conversion was cancelled.');
    }

    await withDeadline(new Promise(resolve => window.setTimeout(resolve, MINERU_POLL_INTERVAL_MS)), deadline, signal);
  }

  if (!fileId) {
    throw new MineruPdfError('MinerU conversion timed out after 30 minutes.');
  }

  onPhase?.('downloading');
  const contentResponse = await withDeadline(requestUrl({
    url: `${v1BaseUrl}/files/${encodeURIComponent(fileId)}/content`,
    method: 'GET',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    throw: false,
  }), deadline, signal);

  if (contentResponse.status < 200 || contentResponse.status >= 300) {
    throw new MineruPdfError(`MinerU result download failed with HTTP ${contentResponse.status}.`);
  }

  return { status: 'ok', markdown: contentResponse.text };
}

async function convertWithLegacyFileParse(
  bytes: Uint8Array,
  filename: string,
  baseUrl: string,
  token: string | undefined,
  deadline: number,
  signal?: AbortSignal,
  onPhase?: (phase: MineruPhase) => void,
): Promise<string> {
  throwIfAborted(signal);
  onPhase?.('uploading');

  const boundary = `----MinerUFormBoundary${Math.random().toString(36).slice(2)}${Date.now()}`;
  const bodyBytes = buildMultipartFormData(
    boundary,
    { return_md: 'true' },
    { name: filename, bytes },
  );

  const body = bodyBytes.byteOffset === 0 && bodyBytes.byteLength === bodyBytes.buffer.byteLength
    ? bodyBytes.buffer as ArrayBuffer
    : bodyBytes.buffer.slice(bodyBytes.byteOffset, bodyBytes.byteOffset + bodyBytes.byteLength) as ArrayBuffer;

  const headers: Record<string, string> = {
    'Content-Type': `multipart/form-data; boundary=${boundary}`,
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  onPhase?.('waiting');
  const response = await withDeadline(requestUrl({
    url: `${baseUrl}/file_parse`,
    method: 'POST',
    headers,
    body,
    throw: false,
  }), deadline, signal);

  if (response.status < 200 || response.status >= 300) {
    throw new MineruPdfError(`MinerU request failed with HTTP ${response.status}.`);
  }

  return extractMarkdownFromMineruApiResponse(response.json, filename);
}

async function convertWithSelfHostedMineru(
  bytes: Uint8Array,
  filename: string,
  baseUrl: string,
  token: string | undefined,
  deadline: number,
  signal?: AbortSignal,
  onPhase?: (phase: MineruPhase) => void,
): Promise<string> {
  const cleanBase = baseUrl.replace(/\/+$/, '');
  const v1Base = cleanBase.endsWith('/v1') ? cleanBase : `${cleanBase}/v1`;
  const legacyBase = cleanBase.replace(/\/v1$/, '');

  const v1Result = await convertWithMineruV1(bytes, filename, v1Base, token, deadline, signal, onPhase);
  if (v1Result.status === 'ok') {
    return v1Result.markdown;
  }

  return convertWithLegacyFileParse(bytes, filename, legacyBase, token, deadline, signal, onPhase);
}

function validateRemoteUrl(value: string, baseUrl: string = MINERU_API_BASE_URL): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new MineruPdfError('MinerU returned an invalid URL.');
  }
  if (url.username || url.password) {
    throw new MineruPdfError('MinerU upload and download URLs must not contain credentials.');
  }
  const isCustomBaseUrl = baseUrl !== MINERU_API_BASE_URL;
  if (!isCustomBaseUrl) {
    const host = url.hostname.toLowerCase();
    const localHost = host === 'localhost' || host.endsWith('.localhost') || host === '[::1]' ||
      /^(?:127\.|10\.|169\.254\.|192\.168\.)/.test(host) ||
      /^172\.(?:1[6-9]|2\d|3[01])\./.test(host);
    if (url.protocol !== 'https:' || localHost) {
      throw new MineruPdfError('MinerU upload and download URLs must be safe HTTPS URLs.');
    }
  } else {
    if (url.protocol !== 'https:' && url.protocol !== 'http:') {
      throw new MineruPdfError('MinerU upload and download URLs must be HTTP or HTTPS.');
    }
  }
  return url.href;
}

export async function convertPdfWithMineru(ctx: PdfConversionContext): Promise<ConversionResult> {
  const baseUrl = resolveMineruBaseUrl(ctx.mineruApiBaseUrl ?? ctx.settings.mineruApiBaseUrl);
  const isSelfHosted = isSelfHostedMineruApi(baseUrl);

  const token = ctx.mineruApiToken?.trim();
  if (!token && !isSelfHosted) throw new MineruPdfError('A MinerU API token is required.');

  // Fast-fail against the file's cached `stat.size` before paying the
  // IO of `readBinary`. The byte-length check below is the authoritative
  // guard (and runs for both fast-fail-miss and cache-hit-miss paths);
  // the stat check is the no-IO path for files we've already cataloged.
  // Two checks, one shared message — defined as a constant so the wording
  // stays in lockstep if the MinerU limit changes.
  const oversizedMessage = 'MinerU accepts files up to 200 MB.';
  if (ctx.pdfFile.stat?.size !== undefined && ctx.pdfFile.stat.size > MINERU_MAX_PDF_BYTES) {
    throw new MineruPdfError(oversizedMessage, 'size-limit');
  }

  const bytes = new Uint8Array(await ctx.app.vault.adapter.readBinary(ctx.pdfFile.path));
  if (bytes.byteLength > MINERU_MAX_PDF_BYTES) {
    throw new MineruPdfError(oversizedMessage, 'size-limit');
  }

  // MinerU model version: hardcoded to 'vlm' (PR #404 default; 'pipeline'
  // and 'MinerU-HTML' are exposed by the API but the vlm model is the
  // recommended path per https://mineru.net/apiManage/docs). When the
  // surface area is clear, this becomes a settings field; for now keeping
  // it inline is consistent with PR #404's existing cache-key shape
  // (`:mineru:vlm:v1`) so existing users do not invalidate cache.
  const modelVersion = 'vlm';

  const sourceHash = await sha256Bytes(bytes, ctx.subtle);
  const cache = createPdfCache(ctx.app);
  const baseUrlKey = baseUrl === MINERU_API_BASE_URL ? '' : `:${baseUrl}`;
  const cacheKey = await hashCacheKey(
    `${sourceHash}:mineru:${modelVersion}:v1${baseUrlKey}`,
    ctx.subtle,
  );
  const cached = await cache.get(cacheKey);
  if (cached) return cached;

  const deadline = Date.now() + MINERU_TIMEOUT_MS;

  if (isSelfHosted) {
    const markdown = await convertWithSelfHostedMineru(
      bytes,
      ctx.pdfFile.name,
      baseUrl,
      token,
      deadline,
      ctx.abortSignal,
      ctx.onMineruPhase,
    );
    const entry: ConversionResult = {
      markdown,
      metadata: {
        convertedAt: new Date().toISOString(),
        converter: `mineru/${modelVersion}`,
      },
    };
    await cache.set(cacheKey, entry);
    return entry;
  }

  if (!token) throw new MineruPdfError('A MinerU API token is required.');

  const lease = await requestUpload(token, ctx.pdfFile.name, modelVersion, deadline, ctx.abortSignal, baseUrl);
  ctx.onMineruPhase?.('uploading');
  await uploadPdf(lease.uploadUrl, bytes, deadline, ctx.abortSignal);
  ctx.onMineruPhase?.('waiting');
  const result = await waitForResult(token, lease.taskId, deadline, ctx.abortSignal, baseUrl);
  ctx.onMineruPhase?.('downloading');
  const zipBytes = await downloadResult(result, deadline, ctx.abortSignal);
  if (zipBytes.byteLength > MINERU_MAX_ZIP_BYTES) {
    throw new MineruPdfError('MinerU result archive exceeds the size limit.');
  }

  const markdown = extractMineruMarkdown(zipBytes);
  const entry: ConversionResult = {
    markdown,
    metadata: {
      convertedAt: new Date().toISOString(),
      converter: `mineru/${modelVersion}`,
    },
  };
  await cache.set(cacheKey, entry);
  return entry;
}

async function requestUpload(
  token: string,
  filename: string,
  modelVersion: 'vlm',
  deadline: number,
  signal?: AbortSignal,
  baseUrl: string = MINERU_API_BASE_URL,
): Promise<{ taskId: string; uploadUrl: string }> {
  const envelope = await mineruRequest(token, '/file-urls/batch', {
    method: 'POST',
    body: JSON.stringify({ files: [{ name: filename }], model_version: modelVersion }),
  }, deadline, signal, baseUrl);
  const taskId = stringValue(envelope.data?.batch_id);
  const uploadUrl = Array.isArray(envelope.data?.file_urls)
    ? stringValue(envelope.data.file_urls[0])
    : undefined;
  if (!taskId || !uploadUrl) throw new MineruPdfError('MinerU returned an invalid upload response.');
  return { taskId, uploadUrl: validateRemoteUrl(uploadUrl, baseUrl) };
}

async function uploadPdf(url: string, bytes: Uint8Array, deadline: number, signal?: AbortSignal): Promise<void> {
  throwIfAborted(signal);
  const body = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer as ArrayBuffer
    : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const response = await withDeadline(requestUrl({
    url,
    method: 'PUT',
    body,
    throw: false,
  }), deadline, signal);
  if (response.status < 200 || response.status >= 300) {
    throw new MineruPdfError(`MinerU PDF upload failed with HTTP ${response.status}.`);
  }
}

async function waitForResult(
  token: string,
  taskId: string,
  deadline: number,
  signal?: AbortSignal,
  baseUrl: string = MINERU_API_BASE_URL,
): Promise<string> {
  while (Date.now() < deadline) {
    throwIfAborted(signal);
    const envelope = await mineruRequest(
      token,
      `/extract-results/batch/${encodeURIComponent(taskId)}`,
      { method: 'GET' },
      deadline,
      signal,
      baseUrl,
    );
    const results = envelope.data?.extract_result;
    const record = Array.isArray(results) && results.length === 1 && typeof results[0] === 'object'
      ? results[0] as Record<string, unknown>
      : undefined;
    const state = stringValue(record?.state);
    if (state === 'done') {
      const zipUrl = stringValue(record?.full_zip_url);
      if (!zipUrl) throw new MineruPdfError('MinerU returned no result archive URL.');
      return validateRemoteUrl(zipUrl, baseUrl);
    }
    if (state === 'failed') {
      const errMsg = stringValue(record?.err_msg);
      throw new MineruPdfError(errMsg ?? 'MinerU conversion failed.', classifyMineruFailure(errMsg));
    }
    if (!state || !['waiting-file', 'pending', 'running', 'converting'].includes(state)) {
      throw new MineruPdfError('MinerU returned an invalid task status.');
    }
    await withDeadline(new Promise(resolve => window.setTimeout(resolve, MINERU_POLL_INTERVAL_MS)), deadline, signal);
  }
  throw new MineruPdfError('MinerU conversion timed out after 30 minutes.');
}

async function downloadResult(url: string, deadline: number, signal?: AbortSignal): Promise<Uint8Array> {
  throwIfAborted(signal);
  const response = await withDeadline(requestUrl({ url, method: 'GET', throw: false }), deadline, signal);
  if (response.status < 200 || response.status >= 300) {
    throw new MineruPdfError(`MinerU result download failed with HTTP ${response.status}.`);
  }
  return new Uint8Array(response.arrayBuffer);
}

async function mineruRequest(
  token: string,
  path: string,
  request: { method: string; body?: string },
  deadline: number,
  signal?: AbortSignal,
  baseUrl: string = MINERU_API_BASE_URL,
): Promise<MineruEnvelope> {
  throwIfAborted(signal);
  const response = await withDeadline(requestUrl({
    url: `${baseUrl}${path}`,
    method: request.method,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(request.body ? { 'Content-Type': 'application/json' } : {}),
    },
    ...(request.body ? { body: request.body } : {}),
    throw: false,
  }), deadline, signal);
  if (response.status < 200 || response.status >= 300) {
    throw new MineruPdfError(`MinerU request failed with HTTP ${response.status}.`);
  }
  const envelope = response.json as MineruEnvelope;
  if (!envelope || (envelope.code !== 0 && envelope.code !== '0')) {
    throw new MineruPdfError(envelope?.msg ?? 'MinerU API request failed.');
  }
  return envelope;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/**
 * Map a server-side task-failure message to a limit code so the engine can
 * show a localized rejection instead of the raw English err_msg. Only the
 * observed page-cap phrasing is classified — unknown messages stay uncoded
 * and keep the pass-through error semantics.
 */
function classifyMineruFailure(message: string | undefined): MineruErrorCode | undefined {
  if (!message) return undefined;
  if (/pages?\s+exceeds\s+limit/i.test(message)) return 'page-limit';
  return undefined;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError();
}
