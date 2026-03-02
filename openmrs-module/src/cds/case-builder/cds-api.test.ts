/**
 * cds-api.test.ts
 *
 * Tests for the CDS API layer: URL guard, HTTP wrapper, SSE streaming parser,
 * CSRF header injection, and health check. No real network calls are made.
 */

import { generateCds, generateCdsStreaming, checkCdsHealth } from './cds-api';
import type { CdsResult, StreamEvent } from './cds-api';

// ─── Helpers ────────────────────────────────────────────────────────────────

const SAFE_HTTPS = 'https://middleware.example.com';
const SAFE_RELATIVE = '/clinicdx-api';
const UNSAFE_HTTP = 'http://middleware.example.com';
const UNSAFE_FTP = 'ftp://middleware.example.com';

function makeReadableStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index++]));
      } else {
        controller.close();
      }
    },
  });
}

function mockOkFetch(bodyStream: ReadableStream<Uint8Array>): void {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    body: bodyStream,
    json: () => Promise.resolve({}),
  } as unknown as Response);
}

function mockJsonFetch(payload: object): void {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    status: 200,
    body: null,
    json: () => Promise.resolve(payload),
  } as unknown as Response);
}

function mockErrorFetch(status: number, detail?: string): void {
  global.fetch = jest.fn().mockResolvedValue({
    ok: false,
    status,
    body: null,
    json: () => Promise.resolve(detail ? { detail } : {}),
  } as unknown as Response);
}

beforeEach(() => {
  // Reset the global fetch mock before every test so call-count from a previous
  // test cannot pollute assertions like `expect(global.fetch).not.toHaveBeenCalled()`.
  (global as any).fetch = jest.fn();
});

afterEach(() => {
  jest.restoreAllMocks();
});

// ─── assertSafeUrl (tested via public API) ────────────────────────────────────

describe('URL safety guard', () => {
  test('accepts https:// URL', async () => {
    mockJsonFetch({ response: 'ok', raw_output: '', kb_queries: [], turns: 1, model_server: 'x' } as CdsResult);
    await expect(generateCds(SAFE_HTTPS, 'prompt')).resolves.toBeDefined();
  });

  test('accepts relative / URL', async () => {
    mockJsonFetch({ response: 'ok', raw_output: '', kb_queries: [], turns: 1, model_server: 'x' } as CdsResult);
    await expect(generateCds(SAFE_RELATIVE, 'prompt')).resolves.toBeDefined();
  });

  test('accepts URL matching window.location.origin', async () => {
    mockJsonFetch({ response: 'ok', raw_output: '', kb_queries: [], turns: 1, model_server: 'x' } as CdsResult);
    await expect(generateCds(window.location.origin + '/api', 'prompt')).resolves.toBeDefined();
  });

  test('rejects http:// URL', async () => {
    await expect(generateCds(UNSAFE_HTTP, 'prompt')).rejects.toThrow(
      "middlewareUrl must start with 'https://'",
    );
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('rejects ftp:// URL', async () => {
    await expect(generateCds(UNSAFE_FTP, 'prompt')).rejects.toThrow();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  test('rejects plain hostname without protocol', async () => {
    await expect(generateCds('middleware.example.com/api', 'prompt')).rejects.toThrow();
  });
});

// ─── generateCds ─────────────────────────────────────────────────────────────

describe('generateCds', () => {
  test('POST to /cds/generate with correct body', async () => {
    const result: CdsResult = {
      response: 'Clinical insight text',
      raw_output: 'raw',
      kb_queries: [{ query: 'malaria', score: 42, source: 'who_guidelines' }],
      turns: 2,
      model_server: 'gemma',
    };
    mockJsonFetch(result);
    const out = await generateCds(SAFE_HTTPS, 'test prompt');
    expect(out).toEqual(result);
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[0]).toBe(`${SAFE_HTTPS}/cds/generate`);
    expect(call[1].method).toBe('POST');
    expect(JSON.parse(call[1].body)).toEqual({ prompt: 'test prompt' });
  });

  test('sends CSRF header', async () => {
    mockJsonFetch({});
    await generateCds(SAFE_RELATIVE, 'p').catch(() => {});
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[1].headers['X-Requested-With']).toBe('XMLHttpRequest');
  });

  test('throws with server detail message on error', async () => {
    mockErrorFetch(422, 'Prompt too long');
    await expect(generateCds(SAFE_HTTPS, 'p')).rejects.toThrow('Prompt too long');
  });

  test('throws with status code when no detail in error body', async () => {
    mockErrorFetch(500);
    await expect(generateCds(SAFE_HTTPS, 'p')).rejects.toThrow('CDS server error (500)');
  });
});

// ─── generateCdsStreaming ─────────────────────────────────────────────────────

describe('generateCdsStreaming', () => {
  test('parses turn_start, token, kb_result, done events', async () => {
    const sseLines = [
      'data: {"type":"turn_start","turn":1}\n',
      'data: {"type":"token","text":"Hello"}\n',
      'data: {"type":"kb_result","query":"malaria","score":55,"source":"who"}\n',
      'data: {"type":"done","turns":2}\n',
    ];
    mockOkFetch(makeReadableStream(sseLines));

    const events: StreamEvent[] = [];
    await generateCdsStreaming(SAFE_RELATIVE, 'prompt', (e) => events.push(e));

    expect(events[0]).toEqual({ type: 'turn_start', turn: 1 });
    expect(events[1]).toEqual({ type: 'token', text: 'Hello' });
    expect(events[2]).toEqual({ type: 'kb_result', query: 'malaria', score: 55, source: 'who' });
    expect(events[3]).toEqual({ type: 'done', turns: 2 });
  });

  test('stops processing on [DONE] sentinel', async () => {
    const sseLines = [
      'data: {"type":"token","text":"A"}\n',
      'data: [DONE]\n',
      'data: {"type":"token","text":"SHOULD_NOT_APPEAR"}\n',
    ];
    mockOkFetch(makeReadableStream(sseLines));

    const texts: string[] = [];
    await generateCdsStreaming(SAFE_RELATIVE, 'prompt', (e) => {
      if (e.type === 'token') texts.push(e.text ?? '');
    });

    expect(texts).toEqual(['A']);
  });

  test('C-3 regression: processes buffer tail with no trailing newline', async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"type":"token","text":"X"}\n'));
        // tail event — no trailing newline (previously discarded)
        controller.enqueue(encoder.encode('data: {"type":"done","turns":3}'));
        controller.close();
      },
    });
    mockOkFetch(body);

    const events: StreamEvent[] = [];
    await generateCdsStreaming(SAFE_RELATIVE, 'prompt', (e) => events.push(e));

    const done = events.find((e) => e.type === 'done');
    expect(done).toBeDefined();
    expect(done?.turns).toBe(3);
  });

  test('skips malformed JSON lines without throwing', async () => {
    const sseLines = [
      'data: {"type":"token","text":"Good"}\n',
      'data: NOT_VALID_JSON\n',
      'data: {"type":"done","turns":1}\n',
    ];
    mockOkFetch(makeReadableStream(sseLines));

    const events: StreamEvent[] = [];
    await expect(
      generateCdsStreaming(SAFE_RELATIVE, 'prompt', (e) => events.push(e)),
    ).resolves.toBeUndefined();
    expect(events.map((e) => e.type)).toEqual(['token', 'done']);
  });

  test('skips lines not starting with "data: "', async () => {
    const sseLines = [
      ': heartbeat\n',
      'event: ping\n',
      'data: {"type":"token","text":"Y"}\n',
    ];
    mockOkFetch(makeReadableStream(sseLines));

    const events: StreamEvent[] = [];
    await generateCdsStreaming(SAFE_RELATIVE, 'prompt', (e) => events.push(e));
    expect(events).toHaveLength(1);
    expect(events[0].text).toBe('Y');
  });

  test('handles events split across multiple chunks', async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        // Split a single event line across two chunks
        controller.enqueue(encoder.encode('data: {"type":"token"'));
        controller.enqueue(encoder.encode(',"text":"Split"}\n'));
        controller.enqueue(encoder.encode('data: {"type":"done","turns":1}\n'));
        controller.close();
      },
    });
    mockOkFetch(body);

    const events: StreamEvent[] = [];
    await generateCdsStreaming(SAFE_RELATIVE, 'prompt', (e) => events.push(e));
    expect(events[0]).toEqual({ type: 'token', text: 'Split' });
  });

  test('throws when response is not ok', async () => {
    mockErrorFetch(503, 'Service unavailable');
    await expect(
      generateCdsStreaming(SAFE_HTTPS, 'prompt', () => {}),
    ).rejects.toThrow('Service unavailable');
  });

  test('propagates AbortError when signal is aborted', async () => {
    global.fetch = jest.fn().mockRejectedValue(
      Object.assign(new Error('AbortError'), { name: 'AbortError' }),
    );
    const controller = new AbortController();
    await expect(
      generateCdsStreaming(SAFE_HTTPS, 'p', () => {}, controller.signal),
    ).rejects.toMatchObject({ name: 'AbortError' });
  });

  test('sends POST to /cds/generate_stream', async () => {
    mockOkFetch(makeReadableStream([]));
    await generateCdsStreaming(SAFE_HTTPS, 'my prompt', () => {}).catch(() => {});
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[0]).toBe(`${SAFE_HTTPS}/cds/generate_stream`);
    expect(call[1].method).toBe('POST');
    expect(call[1].headers['X-Requested-With']).toBe('XMLHttpRequest');
  });

  test('throws when response body is null', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200, body: null, json: () => Promise.resolve({}),
    } as unknown as Response);
    await expect(
      generateCdsStreaming(SAFE_HTTPS, 'p', () => {}),
    ).rejects.toThrow('No response body');
  });
});

// ─── checkCdsHealth ───────────────────────────────────────────────────────────

describe('checkCdsHealth', () => {
  test('returns parsed health object on success', async () => {
    const payload = {
      status: 'ok',
      model_server: { url: 'http://localhost:8080', ok: true },
      kb: { url: 'http://localhost:6333', ok: true },
    };
    mockJsonFetch(payload);
    const result = await checkCdsHealth(SAFE_RELATIVE);
    expect(result.status).toBe('ok');
    expect(result.model_server.ok).toBe(true);
    expect(result.kb.ok).toBe(true);
  });

  test('throws when health check fails', async () => {
    mockErrorFetch(503);
    await expect(checkCdsHealth(SAFE_HTTPS)).rejects.toThrow('Health check failed');
  });

  test('sends CSRF header', async () => {
    mockJsonFetch({ status: 'ok', model_server: { url: '', ok: true }, kb: { url: '', ok: true } });
    await checkCdsHealth(SAFE_RELATIVE);
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[1].headers['X-Requested-With']).toBe('XMLHttpRequest');
  });

  test('rejects unsafe URL', async () => {
    await expect(checkCdsHealth(UNSAFE_HTTP)).rejects.toThrow();
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
