/**
 * case-xml.test.ts
 *
 * Pure-function unit tests for the case text builder.
 * No mocks required — these functions have zero external dependencies.
 * Coverage targets: buildCaseText (all obs types, CIEL lookup, attributes,
 * group members, datetime truncation) and buildModelPrompt (Gemma template).
 */

import { buildCaseText, buildModelPrompt } from './case-xml';
import type { PatientFull, Encounter, Obs } from './openmrs-api';

// ─── Helpers ────────────────────────────────────────────────────────────────

function makePatient(overrides: Partial<PatientFull['person']> = {}, identifiers = [
  { identifier: 'JIN-2024-017872', identifierType: { display: 'OpenMRS ID' } },
]): PatientFull {
  return {
    uuid: 'p-uuid',
    display: 'Test Patient',
    person: {
      uuid: 'person-uuid',
      gender: 'M',
      age: 38,
      birthdate: '1986-08-19T00:00:00.000',
      preferredName: { givenName: 'John', familyName: 'Doe' },
      attributes: [],
      ...overrides,
    },
    identifiers,
  };
}

function makeObs(
  conceptUuid: string,
  conceptDisplay: string,
  datatype: string,
  value: Obs['value'],
  cielCode?: string,
  groupMembers?: Obs[],
): Obs {
  return {
    uuid: `obs-${conceptUuid}`,
    display: `${conceptDisplay}: ${value}`,
    concept: {
      uuid: conceptUuid,
      display: conceptDisplay,
      datatype: { display: datatype },
      conceptClass: { display: 'Finding' },
      mappings: cielCode
        ? [{ conceptReferenceTerm: { code: cielCode, conceptSource: { display: 'CIEL' } } }]
        : [],
    },
    value,
    groupMembers: groupMembers ?? null,
  };
}

function makeEncounter(overrides: Partial<Encounter> = {}): Encounter {
  return {
    uuid: 'enc-1',
    encounterDatetime: '2024-08-18T10:30:00',
    encounterType: { display: 'Mental Health Visit' },
    location: { display: 'Muhimbili National Hospital' },
    obs: [],
    ...overrides,
  };
}

// ─── buildCaseText — header lines ───────────────────────────────────────────

describe('buildCaseText — header', () => {
  test('emits Age, Gender, Birthdate, OpenMRS ID', () => {
    const text = buildCaseText(makePatient(), []);
    expect(text).toMatch('Age: 38.0');
    expect(text).toMatch('Gender: M');
    expect(text).toMatch('Birthdate: 1986-08-19');
    expect(text).toMatch('OpenMRS ID: JIN-2024-017872');
  });

  test('birthdate is truncated to YYYY-MM-DD', () => {
    const text = buildCaseText(makePatient({ birthdate: '1986-08-19T00:00:00.000+0300' }), []);
    expect(text).toMatch('Birthdate: 1986-08-19');
    expect(text).not.toMatch('T');
  });

  test('age uses toFixed(1) formatting', () => {
    const text = buildCaseText(makePatient({ age: 5 }), []);
    expect(text).toMatch('Age: 5.0');
  });

  test('gender uses single character', () => {
    const text = buildCaseText(makePatient({ gender: 'Female' }), []);
    expect(text).toMatch('Gender: F');
  });

  test('falls back to first identifier when no OpenMRS ID', () => {
    const patient = makePatient({}, [
      { identifier: 'NAT-789', identifierType: { display: 'National ID' } },
    ]);
    const text = buildCaseText(patient, []);
    expect(text).toMatch('OpenMRS ID: NAT-789');
  });

  test('uses "Unknown" when identifiers array is empty', () => {
    const patient = makePatient({}, []);
    const text = buildCaseText(patient, []);
    expect(text).toMatch('OpenMRS ID: Unknown');
  });

  test('picks OpenMRS ID over other identifiers', () => {
    const patient = makePatient({}, [
      { identifier: 'NAT-001', identifierType: { display: 'National ID' } },
      { identifier: 'OID-999', identifierType: { display: 'OpenMRS ID' } },
    ]);
    const text = buildCaseText(patient, []);
    expect(text).toMatch('OpenMRS ID: OID-999');
  });
});

// ─── buildCaseText — person attributes ──────────────────────────────────────

describe('buildCaseText — person attributes', () => {
  test('renders string attribute', () => {
    const patient = makePatient({
      attributes: [{
        attributeType: { display: 'Civil Status' },
        display: 'Married',
        value: 'Married',
      }],
    });
    const text = buildCaseText(patient, []);
    expect(text).toMatch('Civil Status: Married');
  });

  test('strips " = " prefix from display value', () => {
    const patient = makePatient({
      attributes: [{
        attributeType: { display: 'Occupation' },
        display: 'Occupation = Farmer',
        value: 'Farmer',
      }],
    });
    const text = buildCaseText(patient, []);
    expect(text).toMatch('Occupation: Farmer');
  });

  test('skips empty attribute values', () => {
    const patient = makePatient({
      attributes: [{
        attributeType: { display: 'Phone Number' },
        display: '',
        value: '',
      }],
    });
    const text = buildCaseText(patient, []);
    expect(text).not.toMatch('Phone Number:');
  });

  test('strips CIEL codes from attribute values', () => {
    const patient = makePatient({
      attributes: [{
        attributeType: { display: 'Civil Status' },
        display: 'Married (CIEL:1057)',
        value: 'Married (CIEL:1057)',
      }],
    });
    const text = buildCaseText(patient, []);
    expect(text).toMatch('Civil Status: Married');
    expect(text).not.toMatch('CIEL:1057');
  });
});

// ─── buildCaseText — encounter header ────────────────────────────────────────

describe('buildCaseText — encounter header', () => {
  test('renders Visit, Date, Location', () => {
    const text = buildCaseText(makePatient(), [makeEncounter()]);
    expect(text).toMatch('Visit: Mental Health Visit');
    expect(text).toMatch('Date: 2024-08-18T10:30:00');
    expect(text).toMatch('Location: Muhimbili National Hospital');
  });

  test('normalises date without T to YYYY-MM-DDTHH:MM:SS', () => {
    const enc = makeEncounter({ encounterDatetime: '2024-08-18' });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Date: 2024-08-18T00:00:00');
  });

  test('truncates datetime longer than 19 chars', () => {
    const enc = makeEncounter({ encounterDatetime: '2024-08-18T10:30:00.000+0300' });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Date: 2024-08-18T10:30:00');
  });

  test('uses "Unknown" when location is null', () => {
    const enc = makeEncounter({ location: null });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Location: Unknown');
  });

  test('multiple encounters all appear in output', () => {
    const enc1 = makeEncounter({ uuid: 'e1', encounterType: { display: 'Vitals' }, encounterDatetime: '2024-01-01T00:00:00' });
    const enc2 = makeEncounter({ uuid: 'e2', encounterType: { display: 'ANC' }, encounterDatetime: '2024-02-01T00:00:00' });
    const text = buildCaseText(makePatient(), [enc1, enc2]);
    expect(text).toMatch('Visit: Vitals');
    expect(text).toMatch('Visit: ANC');
  });
});

// ─── buildCaseText — numeric obs ─────────────────────────────────────────────

describe('buildCaseText — numeric obs', () => {
  test('appends kg unit for weight (CIEL:5089)', () => {
    const enc = makeEncounter({
      obs: [makeObs('5089AAAA', 'Weight (kg)', 'Numeric', 63.2, '5089')],
    });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Weight (kg): 63.2 kg');
  });

  test('appends cmHg unit for systolic BP (CIEL:5085)', () => {
    const enc = makeEncounter({
      obs: [makeObs('5085AAAA', 'Systolic blood pressure', 'Numeric', 120, '5085')],
    });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Systolic blood pressure: 120 mmHg');
  });

  test('no unit suffix when NUMERIC_UNITS value is empty string (CIEL:161439)', () => {
    const enc = makeEncounter({
      obs: [makeObs('161439AAAA', 'Gestational age', 'Numeric', 28, '161439')],
    });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Gestational age: 28');
    // Should NOT have a trailing space or unit
    const line = text.split('\n').find((l) => l.startsWith('Gestational age:'));
    expect(line).toBe('Gestational age: 28');
  });

  test('no unit for obs without a CIEL code', () => {
    const enc = makeEncounter({
      obs: [makeObs('custom-uuid', 'Custom Measurement', 'Numeric', 42)],
    });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Custom Measurement: 42');
  });

  test('skips obs with empty value', () => {
    const enc = makeEncounter({
      obs: [makeObs('5089AAAA', 'Weight (kg)', 'Numeric', null, '5089')],
    });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).not.toMatch('Weight (kg):');
  });
});

// ─── buildCaseText — coded obs ───────────────────────────────────────────────

describe('buildCaseText — coded obs', () => {
  test('renders coded obs using value.display', () => {
    const obs = makeObs('1284AAAA', 'Diagnosis', 'Coded', { display: 'Depression', uuid: 'dep-uuid' }, '1284');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Diagnosis: Depression');
  });

  test('strips CIEL codes from coded answer display', () => {
    const obs = makeObs('1284AAAA', 'Diagnosis', 'Coded', { display: 'Depression (CIEL:135822)', uuid: 'dep-uuid' }, '1284');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).not.toMatch('CIEL:135822');
    expect(text).toMatch('Diagnosis: Depression');
  });

  test('renders coded obs by datatype when no CIEL code', () => {
    const obs = makeObs('custom-uuid', 'Some Coded Field', 'Coded', { display: 'Yes' });
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Some Coded Field: Yes');
  });

  test('handles string value for coded obs (fallback)', () => {
    const obs = makeObs('1284AAAA', 'Diagnosis', 'Coded', 'Hypertension', '1284');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Diagnosis: Hypertension');
  });
});

// ─── buildCaseText — datetime obs ────────────────────────────────────────────

describe('buildCaseText — datetime obs', () => {
  test('truncates datetime value to YYYY-MM-DD (CIEL:5096 return visit date)', () => {
    const obs = makeObs('5096AAAA', 'Return Visit Date', 'Date', '2024-09-15T00:00:00.000', '5096');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Return Visit Date: 2024-09-15');
    // date should NOT include time component
    const line = text.split('\n').find((l) => l.startsWith('Return Visit Date:'));
    expect(line).toBe('Return Visit Date: 2024-09-15');
  });

  test('renders datetime obs by datatype "Date"', () => {
    const obs = makeObs('custom-uuid', 'Appointment Date', 'Date', '2025-01-01T00:00:00');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Appointment Date: 2025-01-01');
  });

  test('renders datetime obs by datatype "Datetime"', () => {
    const obs = makeObs('custom-uuid', 'Onset Time', 'Datetime', '2025-06-15T14:30:00');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Onset Time: 2025-06-15');
  });
});

// ─── buildCaseText — group members ────────────────────────────────────────────

describe('buildCaseText — group members (obs groups)', () => {
  test('flattens group members to leaf obs', () => {
    const child1 = makeObs('5089AAAA', 'Weight (kg)', 'Numeric', 63.2, '5089');
    const child2 = makeObs('5090AAAA', 'Height (cm)', 'Numeric', 170, '5090');
    const group = makeObs('group-uuid', 'Vitals Group', 'N/A', null, undefined, [child1, child2]);
    const enc = makeEncounter({ obs: [group] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Weight (kg): 63.2 kg');
    expect(text).toMatch('Height (cm): 170 cm');
    // group itself should not appear
    expect(text).not.toMatch('Vitals Group:');
  });

  test('deeply nested group members are flattened', () => {
    const leaf = makeObs('5089AAAA', 'Weight (kg)', 'Numeric', 72, '5089');
    const innerGroup = makeObs('ig-uuid', 'Inner Group', 'N/A', null, undefined, [leaf]);
    const outerGroup = makeObs('og-uuid', 'Outer Group', 'N/A', null, undefined, [innerGroup]);
    const enc = makeEncounter({ obs: [outerGroup] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Weight (kg): 72 kg');
  });
});

// ─── buildCaseText — CIEL code extraction ────────────────────────────────────

describe('buildCaseText — CIEL extraction', () => {
  test('recognises Columbia (CIEL) source', () => {
    const obs: Obs = {
      uuid: 'obs-1',
      display: 'Weight: 65',
      concept: {
        uuid: '5089AAAA',
        display: 'Weight (kg)',
        datatype: { display: 'Numeric' },
        conceptClass: { display: 'Finding' },
        mappings: [{
          conceptReferenceTerm: {
            code: '5089',
            conceptSource: { display: 'Columbia International eHealth Laboratory' },
          },
        }],
      },
      value: 65,
      groupMembers: null,
    };
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Weight (kg): 65 kg');
  });

  test('ignores non-CIEL concept sources', () => {
    const obs = makeObs('custom-uuid', 'Custom Lab', 'Numeric', 7.4);
    // add a SNOMED mapping that should not be treated as CIEL
    obs.concept.mappings = [{
      conceptReferenceTerm: { code: '12345', conceptSource: { display: 'SNOMED-CT' } },
    }];
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).toMatch('Custom Lab: 7.4');
    // no unit should be appended (since no CIEL units entry)
  });

  test('stripCiel removes CIEL annotations from concept names', () => {
    const obs = makeObs('5089AAAA', 'Weight (CIEL:5089)', 'Numeric', 50, '5089');
    const enc = makeEncounter({ obs: [obs] });
    const text = buildCaseText(makePatient(), [enc]);
    expect(text).not.toMatch('(CIEL:5089)');
    expect(text).toMatch('Weight: 50 kg');
  });
});

// ─── buildModelPrompt ─────────────────────────────────────────────────────────

describe('buildModelPrompt', () => {
  test('contains Gemma BOS token', () => {
    const p = buildModelPrompt('test case');
    expect(p).toContain('<bos>');
  });

  test('contains user turn start delimiter', () => {
    const p = buildModelPrompt('test case');
    expect(p).toContain('<start_of_turn>user');
  });

  test('contains CDS preamble', () => {
    const p = buildModelPrompt('test case');
    expect(p).toContain('[CDS]');
    expect(p).toContain('evidence-based clinical decision support');
  });

  test('embeds the case text verbatim', () => {
    const caseText = 'Age: 38.0\nGender: M\nDiagnosis: Malaria';
    const p = buildModelPrompt(caseText);
    expect(p).toContain('Age: 38.0');
    expect(p).toContain('Diagnosis: Malaria');
  });

  test('ends with model turn start delimiter', () => {
    const p = buildModelPrompt('test case');
    expect(p.trim()).toMatch(/<start_of_turn>model\s*$/);
  });

  test('contains end_of_turn delimiter', () => {
    const p = buildModelPrompt('test case');
    expect(p).toContain('<end_of_turn>');
  });

  test('user turn precedes model turn', () => {
    const p = buildModelPrompt('test');
    const userIdx = p.indexOf('<start_of_turn>user');
    const modelIdx = p.indexOf('<start_of_turn>model');
    expect(userIdx).toBeLessThan(modelIdx);
  });
});

// ─── output format integrity ──────────────────────────────────────────────────

describe('buildCaseText — output format', () => {
  test('each line follows "Key: Value" pattern', () => {
    const enc = makeEncounter({
      obs: [makeObs('5089AAAA', 'Weight (kg)', 'Numeric', 63.2, '5089')],
    });
    const text = buildCaseText(makePatient(), [enc]);
    const lines = text.split('\n').filter((l) => l.trim());
    for (const line of lines) {
      expect(line).toMatch(/^.+: .+$/);
    }
  });

  test('no XML tags in output', () => {
    const text = buildCaseText(makePatient(), [makeEncounter()]);
    expect(text).not.toMatch(/<[^>]+>/);
  });

  test('empty encounters returns only header lines', () => {
    const text = buildCaseText(makePatient(), []);
    const lines = text.split('\n');
    expect(lines).toHaveLength(4); // Age, Gender, Birthdate, OpenMRS ID
  });
});
