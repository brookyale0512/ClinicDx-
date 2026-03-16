/**
 * cds-workspace.test.tsx
 *
 * Two test suites:
 *   1. Pure unit tests for cleanResponse and parseModelResponse (exported @internal).
 *   2. React component tests for phase rendering, accessibility, and user flows.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { cleanResponse, parseModelResponse } from './cds-workspace.component';
import type { ParsedSection } from './cds-workspace.component';

// ─── Module mocks ────────────────────────────────────────────────────────────

jest.mock('@openmrs/esm-framework', () => ({
  usePatient: jest.fn(),
  useConfig: jest.fn(),
  showSnackbar: jest.fn(),
}));

jest.mock('./case-builder', () => ({
  fetchPatient: jest.fn(),
  fetchEncounters: jest.fn(),
  buildCaseXml: jest.fn().mockReturnValue('Age: 38.0\nGender: M'),
  buildModelPrompt: jest.fn().mockReturnValue('<bos>user\ntest<end_of_turn>\nmodel\n'),
  generateCdsStreaming: jest.fn(),
}));

import { usePatient, useConfig, showSnackbar } from '@openmrs/esm-framework';
import {
  fetchPatient,
  fetchEncounters,
  generateCdsStreaming,
} from './case-builder';
import type { StreamEvent } from './case-builder';

const mockUsePatient = usePatient as jest.MockedFunction<typeof usePatient>;
const mockUseConfig = useConfig as jest.MockedFunction<typeof useConfig>;
const mockFetchPatient = fetchPatient as jest.MockedFunction<typeof fetchPatient>;
const mockFetchEncounters = fetchEncounters as jest.MockedFunction<typeof fetchEncounters>;
const mockStreamingFn = generateCdsStreaming as jest.MockedFunction<typeof generateCdsStreaming>;

const defaultProps = {
  patientUuid: 'test-patient-uuid',
  closeWorkspace: jest.fn(),
  promptBeforeClosing: jest.fn(),
};

const mockPatientData = {
  uuid: 'test-patient-uuid',
  display: 'John Doe',
  person: {
    uuid: 'person-uuid',
    gender: 'M',
    age: 38,
    birthdate: '1986-08-19',
    preferredName: { givenName: 'John', familyName: 'Doe' },
    attributes: [],
  },
  identifiers: [{ identifier: 'OID-001', identifierType: { display: 'OpenMRS ID' } }],
};

function setupDefaultMocks() {
  mockUsePatient.mockReturnValue({ patient: { id: 'test-patient-uuid', name: [{ given: ['John'], family: 'Doe' }], gender: 'M', birthDate: '1986-08-19' } as any, isLoading: false } as any);
  mockUseConfig.mockReturnValue({ middlewareUrl: '/api' } as any);
  mockFetchPatient.mockResolvedValue(mockPatientData as any);
  mockFetchEncounters.mockResolvedValue([]);
  mockStreamingFn.mockResolvedValue(undefined);
}

beforeEach(() => {
  jest.clearAllMocks();
  setupDefaultMocks();
});

// ════════════════════════════════════════════════════════════════════════════
// SECTION 1: Pure unit tests for cleanResponse
// ════════════════════════════════════════════════════════════════════════════

describe('cleanResponse', () => {
  test('removes complete <think>...</think> block', () => {
    const raw = 'Before<think>model reasoning here</think>After';
    expect(cleanResponse(raw)).toBe('BeforeAfter');
  });

  test('removes multiline <think> block', () => {
    const raw = 'Text<think>\nline1\nline2\n</think>More text';
    expect(cleanResponse(raw)).toBe('TextMore text');
  });

  test('removes unclosed <think> block (tail stripping)', () => {
    const raw = 'Good content\n<think>incomplete reasoning that never closes';
    expect(cleanResponse(raw)).toBe('Good content');
  });

  test('removes orphan </think> closing tag', () => {
    const raw = '</think>clean text';
    expect(cleanResponse(raw)).toBe('clean text');
  });

  test('removes <kb_query> blocks', () => {
    const raw = 'Content<kb_query>what is malaria treatment</kb_query>More';
    expect(cleanResponse(raw)).toBe('ContentMore');
  });

  test('removes DECISION: marker', () => {
    const raw = 'Assessment here.\nDECISION: REFER\nMore advice.';
    expect(cleanResponse(raw)).toBe('Assessment here.\n\nMore advice.');
  });

  test('is case-insensitive for think tags', () => {
    const raw = '<THINK>internal</THINK>output';
    expect(cleanResponse(raw)).toBe('output');
  });

  test('trims whitespace from result', () => {
    const raw = '   \nclinical text\n   ';
    expect(cleanResponse(raw)).toBe('clinical text');
  });

  test('handles empty string', () => {
    expect(cleanResponse('')).toBe('');
  });

  test('leaves clean text unchanged', () => {
    const raw = '## Clinical Assessment\n\nPatient has malaria.';
    expect(cleanResponse(raw)).toBe(raw);
  });

  test('multiple think blocks all removed', () => {
    const raw = '<think>a</think>text1<think>b</think>text2';
    expect(cleanResponse(raw)).toBe('text1text2');
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SECTION 2: Pure unit tests for parseModelResponse
// ════════════════════════════════════════════════════════════════════════════

describe('parseModelResponse', () => {
  test('returns empty array for empty string', () => {
    expect(parseModelResponse('')).toEqual([]);
  });

  test('returns empty array for short plain text (<= 20 chars)', () => {
    expect(parseModelResponse('Too short')).toEqual([]);
  });

  test('wraps long plain text (no sections) as Clinical Insight', () => {
    const raw = 'This is a longer clinical response without any markdown sections at all.';
    const sections = parseModelResponse(raw);
    expect(sections).toHaveLength(1);
    expect(sections[0].title).toBe('Clinical Insight');
    expect(sections[0].icon).toBe('report');
    expect(sections[0].content).toContain('clinical response');
  });

  test('parses a single ## section', () => {
    const raw = '## Clinical Assessment\n\nPatient presents with fever and chills.';
    const sections = parseModelResponse(raw);
    expect(sections).toHaveLength(1);
    expect(sections[0].title).toBe('Clinical Assessment');
    expect(sections[0].content).toContain('fever and chills');
  });

  test('parses multiple ## sections', () => {
    const raw = [
      '## Clinical Assessment',
      'Fever noted.',
      '',
      '## Evidence & Considerations',
      'WHO guidelines recommend ACT.',
      '',
      '## Suggested Actions',
      'Start treatment.',
    ].join('\n');
    const sections = parseModelResponse(raw);
    expect(sections).toHaveLength(3);
    expect(sections[0].title).toBe('Clinical Assessment');
    expect(sections[1].title).toBe('Evidence & Considerations');
    expect(sections[2].title).toBe('Suggested Actions');
  });

  test('assigns correct icons based on section title keywords', () => {
    const raw = [
      '## Clinical Assessment\ncontent',
      '## Evidence & Considerations\ncontent',
      '## Suggested Actions\ncontent',
      '## Safety Alert\ncontent',
      '## Key Points\ncontent',
      '## Other Section\ncontent',
    ].join('\n\n');
    const sections = parseModelResponse(raw);
    const iconMap = Object.fromEntries(sections.map((s) => [s.title, s.icon]));
    expect(iconMap['Clinical Assessment']).toBe('stethoscope');
    expect(iconMap['Evidence & Considerations']).toBe('database');
    expect(iconMap['Suggested Actions']).toBe('treatment');
    expect(iconMap['Safety Alert']).toBe('shield');
    expect(iconMap['Key Points']).toBe('list');
    expect(iconMap['Other Section']).toBe('report');
  });

  test('deduplicates sections with the same title (case-insensitive)', () => {
    const raw = [
      '## Clinical Assessment\nFirst occurrence.',
      '## CLINICAL ASSESSMENT\nDuplicate should be ignored.',
    ].join('\n\n');
    const sections = parseModelResponse(raw);
    const caSections = sections.filter((s) =>
      s.title.toLowerCase() === 'clinical assessment',
    );
    expect(caSections).toHaveLength(1);
    expect(caSections[0].content).toContain('First occurrence');
  });

  test('skips sections with empty content', () => {
    const raw = '## Empty Section\n\n## Real Section\nActual content here.';
    const sections = parseModelResponse(raw);
    expect(sections).toHaveLength(1);
    expect(sections[0].title).toBe('Real Section');
  });

  test('strips <think> blocks before parsing sections', () => {
    const raw = '<think>reasoning</think>\n## Clinical Assessment\nClean content.';
    const sections = parseModelResponse(raw);
    expect(sections).toHaveLength(1);
    expect(sections[0].content).toBe('Clean content.');
  });

  test('content does not include the section header itself', () => {
    const raw = '## Clinical Assessment\nPatient is stable.';
    const sections = parseModelResponse(raw);
    expect(sections[0].content).not.toContain('## Clinical Assessment');
    expect(sections[0].content).toContain('Patient is stable.');
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SECTION 3: React component tests
// ════════════════════════════════════════════════════════════════════════════

import CdsWorkspace from './cds-workspace.component';

describe('CdsWorkspace — rendering', () => {
  test('shows loading spinner when patient data is loading', () => {
    mockUsePatient.mockReturnValue({ patient: null, isLoading: true } as any);
    render(<CdsWorkspace {...defaultProps} />);
    expect(screen.getByText('Loading patient data...')).toBeInTheDocument();
  });

  test('shows "Get AI Insight" button in idle phase', () => {
    render(<CdsWorkspace {...defaultProps} />);
    expect(screen.getByText('Get AI Insight')).toBeInTheDocument();
  });

  test('shows patient context tile when patient loaded', () => {
    render(<CdsWorkspace {...defaultProps} />);
    expect(screen.getByText('Patient Context')).toBeInTheDocument();
  });

  test('shows workspace header', () => {
    render(<CdsWorkspace {...defaultProps} />);
    expect(screen.getByText('ClinicDx CDS')).toBeInTheDocument();
    expect(screen.getByText('AI-Powered')).toBeInTheDocument();
  });

  test('renders workspace container', () => {
    render(<CdsWorkspace {...defaultProps} />);
    expect(document.querySelector('[class*="workspace"]')).toBeTruthy();
  });
});

describe('CdsWorkspace — analyze flow', () => {
  test('transitions to "Building case..." text when button clicked', async () => {
    // generateCdsStreaming never resolves — keeps us in building/streaming phase
    mockStreamingFn.mockReturnValue(new Promise(() => {}));
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));
    await waitFor(() => {
      expect(screen.getByText('Building case...')).toBeInTheDocument();
    });
  });

  test('action button is disabled while building', async () => {
    mockStreamingFn.mockReturnValue(new Promise(() => {}));
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));
    await waitFor(() => {
      const btn = screen.getByRole('button', { name: /Building case|Analyzing with AI/i });
      expect(btn).toBeDisabled();
    });
  });

  test('calls generateCdsStreaming with built prompt', async () => {
    mockStreamingFn.mockResolvedValue(undefined);
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));

    await waitFor(() => {
      expect(mockStreamingFn).toHaveBeenCalledTimes(1);
      const [prompt] = mockStreamingFn.mock.calls[0];
      expect(prompt).toContain('<bos>');
    });
  });

  test('transitions to done phase after streaming resolves', async () => {
    mockStreamingFn.mockResolvedValue(undefined);
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));

    await waitFor(() => {
      expect(mockStreamingFn).toHaveBeenCalled();
    });
    // Phase transitions to done (no error) — the "Get AI Insight" button is gone
    await waitFor(() => {
      expect(screen.queryByText('Get AI Insight')).not.toBeInTheDocument();
    });
  });
});

describe('CdsWorkspace — error handling', () => {
  test('shows error tile when fetch throws', async () => {
    mockFetchPatient.mockRejectedValue(new Error('Network failure'));
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));
    await waitFor(() => {
      // The error message appears in both the aria-live region and the error Tile;
      // use getAllByText to handle multiple matching elements.
      expect(screen.getAllByText('Network failure').length).toBeGreaterThan(0);
    });
  });

  test('shows error from stream error event', async () => {
    let capturedOnEvent: ((e: StreamEvent) => void) | undefined;
    mockStreamingFn.mockImplementation(async (_prompt, onEvent, _signal) => {
      capturedOnEvent = onEvent;
    });

    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));
    await waitFor(() => expect(capturedOnEvent).toBeDefined());

    await act(async () => {
      capturedOnEvent!({ type: 'error', message: 'Model overloaded' });
    });

    await waitFor(() => {
      // The error message appears in both the aria-live region and the error Tile.
      expect(screen.getAllByText('Model overloaded').length).toBeGreaterThan(0);
    });
  });

  test('does not show error for AbortError', async () => {
    const abortErr = Object.assign(new Error('Aborted'), { name: 'AbortError' });
    mockFetchPatient.mockRejectedValue(abortErr);
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));
    // Should remain in idle/building without error
    await waitFor(() => {
      expect(screen.queryByText('Aborted')).not.toBeInTheDocument();
    });
  });
});

describe('CdsWorkspace — promptBeforeClosing', () => {
  test('registers a promptBeforeClosing handler on mount', () => {
    render(<CdsWorkspace {...defaultProps} />);
    expect(defaultProps.promptBeforeClosing).toHaveBeenCalled();
  });

  test('the registered handler returns false when idle', () => {
    let handler: (() => boolean) | undefined;
    defaultProps.promptBeforeClosing.mockImplementation((fn: () => boolean) => {
      handler = fn;
    });
    render(<CdsWorkspace {...defaultProps} />);
    expect(handler?.()).toBe(false);
  });
});

describe('CdsWorkspace — KB evidence callback', () => {
  test('passes onEvent callback to generateCdsStreaming', async () => {
    mockStreamingFn.mockResolvedValue(undefined);
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));

    await waitFor(() => {
      expect(mockStreamingFn).toHaveBeenCalledTimes(1);
      const [, onEvent] = mockStreamingFn.mock.calls[0];
      expect(typeof onEvent).toBe('function');
    });
  });

  test('passes AbortSignal to generateCdsStreaming', async () => {
    mockStreamingFn.mockResolvedValue(undefined);
    render(<CdsWorkspace {...defaultProps} />);
    fireEvent.click(screen.getByText('Get AI Insight'));

    await waitFor(() => {
      expect(mockStreamingFn).toHaveBeenCalledTimes(1);
      const [, , signal] = mockStreamingFn.mock.calls[0];
      expect(signal).toBeInstanceOf(AbortSignal);
    });
  });
});
