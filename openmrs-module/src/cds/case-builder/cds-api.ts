export interface KbQueryResult {
  query: string;
  score: number;
  source: string;
}

export interface CdsResult {
  response: string;
  raw_output: string;
  kb_queries: KbQueryResult[];
  turns: number;
  model_server: string;
}

export interface StreamEvent {
  type: 'turn_start' | 'token' | 'kb_query' | 'kb_result' | 'kb_duplicate' | 'done' | 'error';
  text?: string;
  turn?: number;
  query?: string;
  source?: string;
  score?: number;
  message?: string;
  turns?: number;
  kb_queries?: KbQueryResult[];
}

function assertSafeUrl(middlewareUrl: string): void {
  if (
    !middlewareUrl.startsWith('https://') &&
    !middlewareUrl.startsWith('/') &&
    !middlewareUrl.startsWith(window.location.origin)
  ) {
    throw new Error(
      `ClinicDx: middlewareUrl must start with 'https://', '/', or the current origin. ` +
        `Got: ${middlewareUrl}`,
    );
  }
}

const CSRF_HEADER = { 'X-Requested-With': 'XMLHttpRequest' };

export async function generateCds(middlewareUrl: string, prompt: string): Promise<CdsResult> {
  assertSafeUrl(middlewareUrl);
  const res = await fetch(`${middlewareUrl}/cds/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...CSRF_HEADER },
    body: JSON.stringify({ prompt }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail || `CDS server error (${res.status})`);
  }

  return res.json() as Promise<CdsResult>;
}

export async function generateCdsStreaming(
  middlewareUrl: string,
  prompt: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  assertSafeUrl(middlewareUrl);
  const res = await fetch(`${middlewareUrl}/cds/generate_stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...CSRF_HEADER },
    body: JSON.stringify({ prompt }),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail || `CDS server error (${res.status})`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const payload = trimmed.slice(6);
      if (payload === '[DONE]') return;

      try {
        const event: StreamEvent = JSON.parse(payload) as StreamEvent;
        onEvent(event);
      } catch {
        // skip malformed events
      }
    }
  }

  // Process any remaining buffer content after stream ends (C-3 fix)
  if (buffer.trim()) {
    const trimmed = buffer.trim();
    if (trimmed.startsWith('data: ')) {
      const payload = trimmed.slice(6);
      if (payload !== '[DONE]') {
        try {
          const event: StreamEvent = JSON.parse(payload) as StreamEvent;
          onEvent(event);
        } catch {
          // skip malformed tail event
        }
      }
    }
  }
}

export async function checkCdsHealth(middlewareUrl: string): Promise<{
  status: string;
  model_server: { url: string; ok: boolean };
  kb: { url: string; ok: boolean };
}> {
  assertSafeUrl(middlewareUrl);
  const res = await fetch(`${middlewareUrl}/cds/health`, {
    headers: { ...CSRF_HEADER },
  });
  if (!res.ok) throw new Error('Health check failed');
  return res.json() as Promise<{ status: string; model_server: { url: string; ok: boolean }; kb: { url: string; ok: boolean } }>;
}
