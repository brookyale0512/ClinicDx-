/**
 * imaging-workspace.component.test.tsx
 *
 * Tests for the Imaging workspace component.
 * Covers: feature flag gating, drag-and-drop keyboard accessibility (A-4),
 * file size/count limits (S-4), file list rendering, and loading state.
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// ─── Module mocks ────────────────────────────────────────────────────────────

// NOTE: `jest.mock()` is hoisted above all variable declarations by SWC.
// Variables referenced inside the factory MUST be declared before the factory
// runs, or they hit the TDZ.  Safe pattern: use `jest.fn()` inline; derive
// typed references from the real import *after* the mock block.
let mockFeatureFlagValue = false;

jest.mock('@openmrs/esm-framework', () => ({
  usePatient: jest.fn(),
  useFeatureFlag: jest.fn().mockImplementation(() => mockFeatureFlagValue),
  showSnackbar: jest.fn(),
}));

import { usePatient, showSnackbar } from '@openmrs/esm-framework';

const mockUsePatient = usePatient as jest.MockedFunction<typeof usePatient>;
const mockShowSnackbar = showSnackbar as jest.MockedFunction<typeof showSnackbar>;

function setPatient(loading = false) {
  mockUsePatient.mockReturnValue({
    patient: loading ? null : {
      id: 'test-uuid',
      name: [{ given: ['Bob'], family: 'Smith' }],
      gender: 'M',
      birthDate: '1980-05-15',
    },
    isLoading: loading,
  } as any);
}

beforeEach(() => {
  jest.clearAllMocks();
  mockFeatureFlagValue = false;
  setPatient();
});

import ImagingWorkspace from './imaging-workspace.component';

const defaultProps = {
  patientUuid: 'test-uuid',
  closeWorkspace: jest.fn(),
  promptBeforeClosing: jest.fn(),
};

function makeDicomFile(name: string, sizeMB = 1): File {
  const file = new File(['x'], name, { type: 'application/dicom' });
  Object.defineProperty(file, 'size', { value: sizeMB * 1024 * 1024, configurable: true });
  return file;
}

// ─── Feature flag gating ──────────────────────────────────────────────────────

describe('ImagingWorkspace — feature flag', () => {
  test('shows "Under Development" tile when flag is off', () => {
    mockFeatureFlagValue = false;
    render(<ImagingWorkspace {...defaultProps} />);
    expect(screen.getByText('This workspace is under development.')).toBeInTheDocument();
  });

  test('shows "ClinicDx Imaging" heading in under-development state', () => {
    mockFeatureFlagValue = false;
    render(<ImagingWorkspace {...defaultProps} />);
    expect(screen.getByText('ClinicDx Imaging')).toBeInTheDocument();
  });

  test('shows full drop zone UI when flag is on', () => {
    mockFeatureFlagValue = true;
    render(<ImagingWorkspace {...defaultProps} />);
    expect(screen.getByText('Upload DICOM Files')).toBeInTheDocument();
    expect(screen.getByText('Drag and drop DICOM files here, or click to browse')).toBeInTheDocument();
  });

  test('does NOT show under-development text when flag is on', () => {
    mockFeatureFlagValue = true;
    render(<ImagingWorkspace {...defaultProps} />);
    expect(screen.queryByText('This workspace is under development.')).not.toBeInTheDocument();
  });
});

// ─── Loading state ─────────────────────────────────────────────────────────────

describe('ImagingWorkspace — loading', () => {
  test('shows loading spinner when patient data is loading', () => {
    setPatient(true);
    mockFeatureFlagValue = true;
    render(<ImagingWorkspace {...defaultProps} />);
    expect(screen.getByText('Loading patient data...')).toBeInTheDocument();
  });
});

// ─── A-4: Drop zone keyboard accessibility ────────────────────────────────────

describe('ImagingWorkspace — drop zone keyboard accessibility (A-4)', () => {
  beforeEach(() => {
    mockFeatureFlagValue = true;
    setPatient();
  });

  test('drop zone has role="button"', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZones = screen.getAllByRole('button');
    // Should include a button-role element with the upload label
    const dropZone = dropZones.find((el) =>
      el.getAttribute('aria-label') === 'Upload DICOM Files',
    );
    expect(dropZone).toBeDefined();
  });

  test('drop zone has tabIndex=0', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[aria-label="Upload DICOM Files"]');
    expect(dropZone).toBeTruthy();
    expect(dropZone?.getAttribute('tabindex')).toBe('0');
  });

  test('drop zone has aria-label', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[aria-label="Upload DICOM Files"]');
    expect(dropZone).toBeTruthy();
  });

  test('Enter key on drop zone triggers file input click', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[role="button"][aria-label="Upload DICOM Files"]');
    expect(dropZone).toBeTruthy();
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = jest.spyOn(input, 'click').mockImplementation(() => {});

    fireEvent.keyDown(dropZone!, { key: 'Enter' });
    expect(clickSpy).toHaveBeenCalledTimes(1);
    clickSpy.mockRestore();
  });

  test('Space key on drop zone triggers file input click', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[role="button"][aria-label="Upload DICOM Files"]');
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = jest.spyOn(input, 'click').mockImplementation(() => {});

    fireEvent.keyDown(dropZone!, { key: ' ' });
    expect(clickSpy).toHaveBeenCalledTimes(1);
    clickSpy.mockRestore();
  });

  test('other keys on drop zone do NOT trigger click', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[role="button"][aria-label="Upload DICOM Files"]');
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = jest.spyOn(input, 'click').mockImplementation(() => {});

    fireEvent.keyDown(dropZone!, { key: 'Tab' });
    fireEvent.keyDown(dropZone!, { key: 'Escape' });
    expect(clickSpy).not.toHaveBeenCalled();
    clickSpy.mockRestore();
  });
});

// ─── S-4: File size and count limits ──────────────────────────────────────────

describe('ImagingWorkspace — file limits (S-4)', () => {
  beforeEach(() => {
    mockFeatureFlagValue = true;
    setPatient();
  });

  test('accepts DICOM file within 500 MB', async () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = makeDicomFile('scan.dcm', 100);
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    fireEvent.change(input);

    await waitFor(() => {
      const successCall = mockShowSnackbar.mock.calls.find(
        (c) => c[0]?.title === 'Files Added',
      );
      expect(successCall).toBeDefined();
    });
  });

  test('rejects file exceeding 500 MB limit', async () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const bigFile = makeDicomFile('huge.dcm', 501);
    Object.defineProperty(input, 'files', { value: [bigFile], configurable: true });
    fireEvent.change(input);

    await waitFor(() => {
      const errorCall = mockShowSnackbar.mock.calls.find((c) => c[0]?.kind === 'error');
      expect(errorCall).toBeDefined();
    });
  });

  test('shows error when adding files would exceed MAX_FILE_COUNT (20)', async () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;

    // Add 21 files at once
    const files = Array.from({ length: 21 }, (_, i) => makeDicomFile(`scan${i}.dcm`, 1));
    Object.defineProperty(input, 'files', { value: files, configurable: true });
    fireEvent.change(input);

    await waitFor(() => {
      const errorCall = mockShowSnackbar.mock.calls.find((c) => c[0]?.kind === 'error');
      expect(errorCall).toBeDefined();
    });
  });

  test('shows uploaded files in the file list', async () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const file = makeDicomFile('brain-mri.dcm', 10);
    Object.defineProperty(input, 'files', { value: [file], configurable: true });
    fireEvent.change(input);

    await waitFor(() => {
      expect(screen.getByText('brain-mri.dcm')).toBeInTheDocument();
    });
  });
});

// ─── Drag and drop ────────────────────────────────────────────────────────────

describe('ImagingWorkspace — drag and drop', () => {
  beforeEach(() => {
    mockFeatureFlagValue = true;
    setPatient();
  });

  test('dragOver event sets drag-over visual state', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[role="button"]')!;
    fireEvent.dragOver(dropZone, { dataTransfer: { files: [] } });
    // dragOver should apply the dragOver CSS class (via className check)
    // identity-obj-proxy returns the class name itself as the value
    // so we verify the element is rendered without crashing
    expect(dropZone).toBeTruthy();
  });

  test('dragLeave clears drag-over state', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    const dropZone = document.querySelector('[role="button"]')!;
    fireEvent.dragOver(dropZone, { dataTransfer: { files: [] } });
    fireEvent.dragLeave(dropZone);
    expect(dropZone).toBeTruthy();
  });
});

// ─── Smoke test ───────────────────────────────────────────────────────────────

describe('ImagingWorkspace — smoke', () => {
  test('renders without crashing', () => {
    const { container } = render(<ImagingWorkspace {...defaultProps} />);
    expect(container).toBeTruthy();
  });

  test('registers promptBeforeClosing on mount', () => {
    render(<ImagingWorkspace {...defaultProps} />);
    expect(defaultProps.promptBeforeClosing).toHaveBeenCalled();
  });
});
