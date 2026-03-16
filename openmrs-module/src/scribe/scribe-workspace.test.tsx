/**
 * scribe-workspace.test.tsx
 *
 * Two suites:
 *   1. Pure unit tests for parseItems and handleExtractedData.
 *   2. React component smoke tests for phase rendering and encounter selector.
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { parseItems, handleExtractedData } from './scribe-workspace.component';

// ─── Module mocks ────────────────────────────────────────────────────────────

jest.mock('@openmrs/esm-framework', () => ({
  usePatient: jest.fn().mockReturnValue({ patient: null, isLoading: false }),
  useVisit: jest.fn().mockReturnValue({ activeVisit: null }),
  useConfig: jest.fn().mockReturnValue({
    middlewareUrl: '/api',
    vitalsEncounterTypeUuid: '67a71486-1a54-468f-ac3e-7091a9a79584',
  }),
  openmrsFetch: jest.fn().mockResolvedValue({ data: { results: [] } }),
  restBaseUrl: '/ws/rest/v1',
  showSnackbar: jest.fn(),
}));

// ════════════════════════════════════════════════════════════════════════════
// SECTION 1: parseItems
// ════════════════════════════════════════════════════════════════════════════

describe('parseItems', () => {
  test('returns empty array for empty string', () => {
    expect(parseItems('')).toEqual([]);
  });

  test('returns empty array for whitespace-only input', () => {
    expect(parseItems('   \n\n   ')).toEqual([]);
  });

  test('parses a simple "Key: Value" line', () => {
    const items = parseItems('Weight: 63.2 kg');
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Weight');
    expect(items[0].value).toBe('63.2 kg');
    expect(items[0].checked).toBe(true);
  });

  test('preserves colons within the value', () => {
    const items = parseItems('BP: 120/80 mmHg');
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('BP');
    expect(items[0].value).toBe('120/80 mmHg');
  });

  test('strips complete <think>...</think> block', () => {
    const raw = '<think>model reasoning</think>\nWeight: 63.2 kg';
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Weight');
  });

  test('strips unclosed <think> tail', () => {
    const raw = 'Height: 170 cm\n<think>never closed';
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Height');
  });

  test('strips leading bullet "- " from lines', () => {
    const items = parseItems('- Weight: 63.2 kg');
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Weight');
  });

  test('strips leading bullet "* " from lines', () => {
    const items = parseItems('* Height: 170 cm');
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Height');
  });

  test('skips lines without colon separator', () => {
    const raw = 'no colon here\nWeight: 63.2 kg';
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Weight');
  });

  test.each([
    ['i need'],
    ['i will'],
    ['let me'],
    ['the patient'],
    ['observations'],
    ['concept'],
    ['extract'],
    ['here'],
    ['based on'],
    ['from the'],
    ['audio'],
    ['manifest'],
    ['note'],
    ['output'],
    ['result'],
  ])('skips SKIP_PREFIX line starting with "%s"', (prefix) => {
    const raw = `${prefix}: something irrelevant\nWeight: 70 kg`;
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Weight');
  });

  test('skips labels longer than 50 characters', () => {
    const longLabel = 'a'.repeat(51);
    const raw = `${longLabel}: some value\nWeight: 70 kg`;
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Weight');
  });

  test('skips labels with more than 5 words', () => {
    const raw = 'this is a six word label: value\nHeight: 170 cm';
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Height');
  });

  test('human_readable uses space-separated label', () => {
    const items = parseItems('blood_pressure: 120/80');
    expect(items[0].human_readable).toBe('blood pressure: 120/80');
  });

  test('assigns sequential item IDs', () => {
    const raw = 'Weight: 63.2 kg\nHeight: 170 cm\nTemp: 37.5 C';
    const items = parseItems(raw);
    expect(items[0].id).toBe('item-0');
    expect(items[1].id).toBe('item-1');
    expect(items[2].id).toBe('item-2');
  });

  test('all items default to checked=true', () => {
    const raw = 'Weight: 63.2 kg\nHeight: 170 cm';
    const items = parseItems(raw);
    expect(items.every((i) => i.checked)).toBe(true);
  });

  test('multiple observations parsed from multiline output', () => {
    const raw = [
      'Weight: 63.2 kg',
      'Height: 170 cm',
      'Temperature: 37.5 C',
      'Blood Pressure: 120/80 mmHg',
    ].join('\n');
    const items = parseItems(raw);
    expect(items).toHaveLength(4);
  });

  test('skips empty label or value', () => {
    const raw = ': no label\nno value:\nGood: value';
    const items = parseItems(raw);
    expect(items).toHaveLength(1);
    expect(items[0].label).toBe('Good');
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SECTION 2: handleExtractedData
// ════════════════════════════════════════════════════════════════════════════

describe('handleExtractedData', () => {
  test('returns structured items when items array is present', () => {
    const data = {
      items: [
        { label: 'Weight', value: '63.2 kg', human_readable: 'Weight: 63.2 kg' },
        { label: 'Height', value: '170 cm' },
      ],
    };
    const result = handleExtractedData(data);
    expect(result).toHaveLength(2);
    expect(result[0].label).toBe('Weight');
    expect(result[1].label).toBe('Height');
  });

  test('structured items default checked=true', () => {
    const data = { items: [{ label: 'Temp', value: '37.5' }] };
    const result = handleExtractedData(data);
    expect(result[0].checked).toBe(true);
  });

  test('structured items preserve human_readable', () => {
    const data = {
      items: [{ label: 'Weight', value: '63.2 kg', human_readable: 'Weight: 63.2 kg' }],
    };
    const result = handleExtractedData(data);
    expect(result[0].human_readable).toBe('Weight: 63.2 kg');
  });

  test('generates human_readable fallback when not provided', () => {
    const data = { items: [{ label: 'BP', value: '120/80' }] };
    const result = handleExtractedData(data);
    expect(result[0].human_readable).toBe('BP: 120/80');
  });

  test('structured items preserve fhir_type and fhir_payload', () => {
    const payload = { resourceType: 'Observation', status: 'final' };
    const data = {
      items: [{ label: 'Glucose', value: '5.4', fhir_type: 'Observation', fhir_payload: payload }],
    };
    const result = handleExtractedData(data);
    expect(result[0].fhir_type).toBe('Observation');
    expect(result[0].fhir_payload).toEqual(payload);
  });

  test('falls back to raw_model_output when no items array', () => {
    const data = { raw_model_output: 'Weight: 70 kg\nHeight: 175 cm' };
    const result = handleExtractedData(data);
    expect(result.length).toBeGreaterThan(0);
    expect(result.some((i) => i.label === 'Weight')).toBe(true);
  });

  test('falls back to raw_output when no items and no raw_model_output', () => {
    const data = { raw_output: 'BP: 120/80' };
    const result = handleExtractedData(data);
    expect(result).toHaveLength(1);
    expect(result[0].label).toBe('BP');
  });

  test('returns empty array when all fields are missing', () => {
    const result = handleExtractedData({});
    expect(result).toEqual([]);
  });

  test('prefers items array over raw_model_output when both present', () => {
    const data = {
      items: [{ label: 'FromItems', value: '1' }],
      raw_model_output: 'FromRaw: value',
    };
    const result = handleExtractedData(data);
    expect(result).toHaveLength(1);
    expect(result[0].label).toBe('FromItems');
  });
});

// ════════════════════════════════════════════════════════════════════════════
// SECTION 3: Component rendering
// ════════════════════════════════════════════════════════════════════════════

import ScribeWorkspace from './scribe-workspace.component';

const defaultProps = {
  patientUuid: 'test-uuid',
  closeWorkspace: jest.fn(),
  promptBeforeClosing: jest.fn(),
};

describe('ScribeWorkspace — rendering', () => {
  // Reset usePatient before each test so a sticky mockReturnValue from one
  // test cannot bleed into the next test.
  beforeEach(() => {
    const { usePatient } = require('@openmrs/esm-framework');
    usePatient.mockReturnValue({ patient: null, isLoading: false });
  });

  test('renders without crashing (smoke test)', () => {
    const { container } = render(<ScribeWorkspace {...defaultProps} />);
    expect(container).toBeTruthy();
  });

  test('shows header with ClinicDx Scribe', () => {
    render(<ScribeWorkspace {...defaultProps} />);
    expect(screen.getByText('ClinicDx Scribe')).toBeInTheDocument();
  });

  test('shows loading spinner when patient is loading', () => {
    const { usePatient } = require('@openmrs/esm-framework');
    // Use mockReturnValue (persistent) so every re-render triggered by
    // async effects (openmrsFetch resolution) still sees isLoading: true.
    usePatient.mockReturnValue({ patient: null, isLoading: true });
    render(<ScribeWorkspace {...defaultProps} />);
    expect(screen.getByText('Loading patient data...')).toBeInTheDocument();
  });

  test('mic button is disabled when no encounter is selected (encounters empty)', () => {
    render(<ScribeWorkspace {...defaultProps} />);
    // mic button should be disabled since no encounter
    const micBtn = document.querySelector('button[aria-label="Hold to speak"]') as HTMLButtonElement;
    if (micBtn) {
      expect(micBtn).toBeDisabled();
    }
    // The label should show "Select an encounter first"
    expect(screen.getByText('Select an encounter first')).toBeInTheDocument();
  });

  test('renders workspace container', () => {
    render(<ScribeWorkspace {...defaultProps} />);
    expect(document.querySelector('[class*="workspace"]')).toBeTruthy();
  });

  test('registers promptBeforeClosing on mount', () => {
    render(<ScribeWorkspace {...defaultProps} />);
    expect(defaultProps.promptBeforeClosing).toHaveBeenCalled();
  });
});
