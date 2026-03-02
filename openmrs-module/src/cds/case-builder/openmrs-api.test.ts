/**
 * openmrs-api.test.ts
 *
 * Tests for the OpenMRS REST API fetch helpers.
 * Verifies URL construction, representation parameters, and response mapping.
 */

import { fetchPatient, fetchEncounters } from './openmrs-api';

// Mock the framework at module level
jest.mock('@openmrs/esm-framework', () => ({
  openmrsFetch: jest.fn(),
  restBaseUrl: '/ws/rest/v1',
}));

import { openmrsFetch } from '@openmrs/esm-framework';

const mockFetch = openmrsFetch as jest.MockedFunction<typeof openmrsFetch>;

afterEach(() => {
  mockFetch.mockReset();
});

// ─── fetchPatient ─────────────────────────────────────────────────────────────

describe('fetchPatient', () => {
  const mockPatient = {
    uuid: 'p-uuid',
    display: 'Test Patient',
    person: {
      uuid: 'person-uuid',
      gender: 'M',
      age: 38,
      birthdate: '1986-08-19T00:00:00.000',
      preferredName: { givenName: 'John', familyName: 'Doe' },
      attributes: [],
    },
    identifiers: [
      { identifier: 'JIN-2024-017872', identifierType: { display: 'OpenMRS ID' } },
    ],
  };

  test('calls openmrsFetch with patient UUID in URL', async () => {
    mockFetch.mockResolvedValue({ data: mockPatient } as any);
    await fetchPatient('p-uuid');
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('/patient/p-uuid');
    expect(url).toContain('/ws/rest/v1');
  });

  test('includes custom representation in URL', async () => {
    mockFetch.mockResolvedValue({ data: mockPatient } as any);
    await fetchPatient('p-uuid');
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('v=custom:');
    expect(url).toContain('identifiers');
    expect(url).toContain('preferredName');
    expect(url).toContain('attributes');
  });

  test('returns the patient data from response', async () => {
    mockFetch.mockResolvedValue({ data: mockPatient } as any);
    const result = await fetchPatient('p-uuid');
    expect(result).toEqual(mockPatient);
  });

  test('propagates fetch errors', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));
    await expect(fetchPatient('p-uuid')).rejects.toThrow('Network error');
  });
});

// ─── fetchEncounters ──────────────────────────────────────────────────────────

describe('fetchEncounters', () => {
  const mockEncounters = [
    {
      uuid: 'enc-1',
      encounterDatetime: '2024-01-01T00:00:00',
      encounterType: { display: 'Vitals' },
      location: { display: 'Ward A' },
      obs: [],
    },
    {
      uuid: 'enc-2',
      encounterDatetime: '2024-02-01T00:00:00',
      encounterType: { display: 'ANC' },
      location: { display: 'Ward B' },
      obs: [],
    },
  ];

  test('calls openmrsFetch with patient UUID', async () => {
    mockFetch.mockResolvedValue({ data: { results: mockEncounters } } as any);
    await fetchEncounters('p-uuid');
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('patient=p-uuid');
  });

  test('uses default limit of 50', async () => {
    mockFetch.mockResolvedValue({ data: { results: [] } } as any);
    await fetchEncounters('p-uuid');
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('limit=50');
  });

  test('uses custom limit when provided', async () => {
    mockFetch.mockResolvedValue({ data: { results: [] } } as any);
    await fetchEncounters('p-uuid', 10);
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('limit=10');
  });

  test('requests ascending order from API', async () => {
    mockFetch.mockResolvedValue({ data: { results: [] } } as any);
    await fetchEncounters('p-uuid');
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('order=asc');
  });

  test('includes obs representation with CIEL mappings', async () => {
    mockFetch.mockResolvedValue({ data: { results: [] } } as any);
    await fetchEncounters('p-uuid');
    const url: string = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('mappings');
    expect(url).toContain('conceptSource');
  });

  test('returns results array directly (no re-sort)', async () => {
    // API returns in desc order — fetchEncounters must return them AS-IS
    // (the API handles sorting via order=asc param)
    const reversed = [...mockEncounters].reverse();
    mockFetch.mockResolvedValue({ data: { results: reversed } } as any);
    const result = await fetchEncounters('p-uuid');
    expect(result[0].uuid).toBe(reversed[0].uuid);
    expect(result[1].uuid).toBe(reversed[1].uuid);
  });

  test('returns empty array when results is missing', async () => {
    mockFetch.mockResolvedValue({ data: {} } as any);
    const result = await fetchEncounters('p-uuid');
    expect(result).toEqual([]);
  });

  test('propagates fetch errors', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));
    await expect(fetchEncounters('p-uuid')).rejects.toThrow('Network error');
  });
});
