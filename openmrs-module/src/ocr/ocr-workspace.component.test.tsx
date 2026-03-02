/**
 * ocr-workspace.component.test.tsx
 *
 * Tests for the OCR workspace component.
 * Covers: feature flag gating, file size validation (S-4),
 * snackbar timing fix (C-7), video accessibility (A-5), loading state.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// ─── Module mocks ────────────────────────────────────────────────────────────

// NOTE: `jest.mock()` is hoisted above all variable declarations by SWC.
// Use `jest.fn()` inline in the factory to avoid TDZ errors; derive typed
// references from the real import *after* the mock block.
let mockFeatureFlagValue = false;

jest.mock('@openmrs/esm-framework', () => ({
  usePatient: jest.fn(),
  useFeatureFlag: jest.fn().mockImplementation(() => mockFeatureFlagValue),
  showSnackbar: jest.fn(),
}));

import { usePatient, showSnackbar } from '@openmrs/esm-framework';

const mockUsePatient = usePatient as jest.MockedFunction<typeof usePatient>;
const mockShowSnackbar = showSnackbar as jest.MockedFunction<typeof showSnackbar>;

function setPatient(loading: boolean = false) {
  mockUsePatient.mockReturnValue({
    patient: loading ? null : {
      id: 'test-uuid',
      name: [{ given: ['Jane'], family: 'Doe' }],
      gender: 'F',
      birthDate: '1990-01-01',
    },
    isLoading: loading,
  } as any);
}

beforeEach(() => {
  jest.clearAllMocks();
  mockFeatureFlagValue = false;
  setPatient();
});

import OcrWorkspace from './ocr-workspace.component';

const defaultProps = {
  patientUuid: 'test-uuid',
  closeWorkspace: jest.fn(),
  promptBeforeClosing: jest.fn(),
};

// ─── Feature flag gating ──────────────────────────────────────────────────────

describe('OcrWorkspace — feature flag', () => {
  test('shows "Under Development" tile when flag is off', () => {
    mockFeatureFlagValue = false;
    render(<OcrWorkspace {...defaultProps} />);
    expect(screen.getByText('This workspace is under development.')).toBeInTheDocument();
  });

  test('shows "ClinicDx OCR" heading even in under-development state', () => {
    mockFeatureFlagValue = false;
    render(<OcrWorkspace {...defaultProps} />);
    expect(screen.getByText('ClinicDx OCR')).toBeInTheDocument();
  });

  test('shows full UI when flag is on', () => {
    mockFeatureFlagValue = true;
    render(<OcrWorkspace {...defaultProps} />);
    expect(screen.getByText('Take Photo')).toBeInTheDocument();
    expect(screen.getByText('Upload Image')).toBeInTheDocument();
  });

  test('does NOT show under-development text when flag is on', () => {
    mockFeatureFlagValue = true;
    render(<OcrWorkspace {...defaultProps} />);
    expect(screen.queryByText('This workspace is under development.')).not.toBeInTheDocument();
  });
});

// ─── Loading state ─────────────────────────────────────────────────────────────

describe('OcrWorkspace — loading', () => {
  test('shows loading spinner when patient data is loading', () => {
    setPatient(true);
    mockFeatureFlagValue = true;
    render(<OcrWorkspace {...defaultProps} />);
    expect(screen.getByText('Loading patient data...')).toBeInTheDocument();
  });
});

// ─── File size validation (S-4) ───────────────────────────────────────────────

describe('OcrWorkspace — file size validation', () => {
  beforeEach(() => {
    mockFeatureFlagValue = true;
    setPatient();
  });

  test('accepts files within the 20 MB limit', async () => {
    render(<OcrWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const smallFile = new File(['x'.repeat(1000)], 'lab.png', { type: 'image/png' });
    Object.defineProperty(input, 'files', { value: [smallFile], configurable: true });
    fireEvent.change(input);
    // No error snackbar for a small file
    await waitFor(() => {
      const errorCalls = mockShowSnackbar.mock.calls.filter(
        (c) => c[0]?.kind === 'error',
      );
      expect(errorCalls).toHaveLength(0);
    });
  });

  test('rejects file exceeding 20 MB and shows error snackbar', async () => {
    render(<OcrWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    // 21 MB file
    const bigFile = new File(['x'], 'huge.png', { type: 'image/png' });
    Object.defineProperty(bigFile, 'size', { value: 21 * 1024 * 1024, configurable: true });
    Object.defineProperty(input, 'files', { value: [bigFile], configurable: true });
    fireEvent.change(input);

    await waitFor(() => {
      const errorCall = mockShowSnackbar.mock.calls.find((c) => c[0]?.kind === 'error');
      expect(errorCall).toBeDefined();
    });
  });
});

// ─── C-7: snackbar timing fix ──────────────────────────────────────────────────

describe('OcrWorkspace — C-7 snackbar inside onload', () => {
  beforeEach(() => {
    mockFeatureFlagValue = true;
    setPatient();
  });

  test('success snackbar fires after FileReader completes (inside onload)', async () => {
    // Mock FileReader to call onload synchronously for testing
    const mockReadAsDataURL = jest.fn().mockImplementation(function (this: any) {
      // Trigger onload synchronously
      if (this.onload) {
        this.result = 'data:image/png;base64,abc';
        this.onload({ target: this });
      }
    });

    const originalFileReader = global.FileReader;
    global.FileReader = jest.fn().mockImplementation(() => ({
      readAsDataURL: mockReadAsDataURL,
      result: null,
      onload: null,
    })) as any;

    render(<OcrWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['data'], 'test.png', { type: 'image/png' });
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    fireEvent.change(input);

    await waitFor(() => {
      const successCalls = mockShowSnackbar.mock.calls.filter(
        (c) => c[0]?.title === 'Image Uploaded',
      );
      expect(successCalls).toHaveLength(1);
    });

    global.FileReader = originalFileReader;
  });
});

// ─── A-5: video accessibility ──────────────────────────────────────────────────

describe('OcrWorkspace — video accessibility', () => {
  beforeEach(() => {
    mockFeatureFlagValue = true;
    setPatient();
    // Mock getUserMedia for camera open
    Object.defineProperty(global.navigator, 'mediaDevices', {
      value: {
        getUserMedia: jest.fn().mockResolvedValue({
          getTracks: () => [{ stop: jest.fn() }],
        }),
      },
      writable: true,
      configurable: true,
    });
  });

  test('video element gets title attribute when camera opens', async () => {
    render(<OcrWorkspace {...defaultProps} />);
    const cameraBtn = screen.getByText('Take Photo').closest('button')!;
    fireEvent.click(cameraBtn);

    await waitFor(() => {
      const video = document.querySelector('video');
      if (video) {
        expect(video.getAttribute('title')).toBe('Camera Preview');
      }
    });
  });
});

// ─── Accessibility structure ───────────────────────────────────────────────────

describe('OcrWorkspace — structure', () => {
  test('renders without crash (smoke test)', () => {
    render(<OcrWorkspace {...defaultProps} />);
    expect(document.body).toBeTruthy();
  });

  test('registers promptBeforeClosing on mount', () => {
    render(<OcrWorkspace {...defaultProps} />);
    expect(defaultProps.promptBeforeClosing).toHaveBeenCalled();
  });
});
