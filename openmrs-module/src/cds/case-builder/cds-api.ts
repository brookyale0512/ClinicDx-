import { openmrsFetch, getConfig } from '@openmrs/esm-framework';

function getMiddlewareUrl(): string {
  try {
    const config = getConfig('@clinicdx/esm-clinicdx-app') as { middlewareUrl?: string };
    if (config?.middlewareUrl) {
      return config.middlewareUrl.replace(/\/$/, '');
    }
  } catch {
    // getConfig throws before config is resolved; fall through to default
  }
  return `${window.location.origin}/clinicdx-api`;
}

export interface KbHit {
  title: string;
  content: string;
  score: number;
  source: string;
  uri: string;
}

export interface KbQueryResult {
  query: string;
  score: number;
  source: string;
  hits: KbHit[];
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
  hits?: KbHit[];
  message?: string;
  turns?: number;
  kb_queries?: KbQueryResult[];
}

export async function generateCds(prompt: string): Promise<CdsResult> {
  const res = await fetch(`${getMiddlewareUrl()}/cds/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `CDS server error (${res.status})`);
  }

  return res.json();
}

export async function generateCdsStreaming(
  prompt: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${getMiddlewareUrl()}/cds/generate_stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `CDS server error (${res.status})`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (!done) {
      buffer += decoder.decode(value, { stream: true });
    }

    const lines = buffer.split('\n');
    buffer = done ? '' : (lines.pop() ?? '');

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const payload = trimmed.slice(6);
      if (payload === '[DONE]') return;

      try {
        const event: StreamEvent = JSON.parse(payload);
        onEvent(event);
      } catch {
        // skip malformed events
      }
    }

    if (done) break;
  }
}

export async function checkCdsHealth(): Promise<{
  status: string;
  model_server: { url: string; ok: boolean };
  kb: { url: string; ok: boolean };
}> {
  const res = await fetch(`${getMiddlewareUrl()}/cds/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
