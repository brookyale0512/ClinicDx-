/**
 * case-text.ts — Build plain-text case representation matching the
 * training data format used for the CDS think model.
 *
 * Training-data canonical form (no XML, no CIEL codes, no system prompt):
 *
 *   [CDS]
 *   Provide evidence-based clinical decision support for this patient.
 *
 *   Age: 38.0
 *   Gender: M
 *   Birthdate: 1986-08-19
 *   OpenMRS ID: JIN-2024-017872
 *   Civil Status: Married
 *   Visit: Mental Health Visit
 *   Date: 2024-08-18T00:00:00
 *   Location: Muhimbili National Hospital
 *   Weight (kg): 63.2 kg
 *   Diagnosis: Depression
 *   Return Visit Date: 2024-09-15
 */

import type { PatientFull, Encounter, Obs, ObsConcept } from './openmrs-api';

// ── CIEL helpers ───────────────────────────────────────────────────

function extractCielCode(concept: ObsConcept): string | null {
  for (const m of concept.mappings ?? []) {
    const src = m.conceptReferenceTerm?.conceptSource?.display ?? '';
    if (src.toUpperCase() === 'CIEL' || src.includes('Columbia')) {
      return m.conceptReferenceTerm.code;
    }
  }
  return null;
}

function stripCiel(name: string): string {
  return name.replace(/\s*\(CIEL:\d+\)/g, '').trim();
}

// ── Units for numeric obs ──────────────────────────────────────────

const NUMERIC_UNITS: Record<string, string> = {
  '5089': 'kg', '5090': 'cm', '5085': 'mmHg', '5086': 'mmHg',
  '5087': 'bpm', '5088': 'C', '5242': 'breaths/min', '5092': '%',
  '21': 'g/dL', '678': '10^3/uL', '887': 'mmol/L', '161439': '',
  '167209': 'cells/hpf', '729': 'cells/uL', '5497': 'cells/uL',
  '856': 'copies/mL', '1342': 'mg/dL', '790': 'mmol/L', '857': 'U/L',
  '1132': 'mmol/L', '1133': 'mmol/L', '1135': 'mmol/L',
  '655': 'umol/L', '1825': 'g/L',
};

// ── Coded obs (render as value_coded) ──────────────────────────────

const CODED_OBS: Set<string> = new Set([
  '1284', '152722', '165268', '120749', '139084', '80', '832',
  '1169', '5272', '299', '5978', '140238', '136443', '143264',
  '5960', '117671', '1560', '374',
]);

// ── Datetime obs ───────────────────────────────────────────────────

const DATETIME_OBS: Set<string> = new Set([
  '5096', '1644', '1427', '164400', '159599', '163526',
]);

// ── Obs rendering ──────────────────────────────────────────────────

type ObsValue = string | number | boolean | { display: string; uuid?: string } | null;

function fmtValue(raw: ObsValue): string {
  if (raw == null) return '';
  if (typeof raw === 'object') return raw.display ?? String(raw);
  return String(raw);
}

function isCoded(ciel: string | null, datatype: string): boolean {
  if (ciel && CODED_OBS.has(ciel)) return true;
  return datatype.toLowerCase() === 'coded';
}

function isDatetime(ciel: string | null, datatype: string): boolean {
  if (ciel && DATETIME_OBS.has(ciel)) return true;
  const dt = datatype.toLowerCase();
  return dt === 'date' || dt === 'datetime';
}

function renderObs(lines: string[], obs: Obs): void {
  if (obs.groupMembers?.length) {
    for (const member of obs.groupMembers) renderObs(lines, member);
    return;
  }

  const concept = obs.concept;
  const name = stripCiel(concept.display ?? 'Unknown');
  const ciel = extractCielCode(concept);
  const datatype = concept.datatype?.display ?? '';

  if (isDatetime(ciel, datatype)) {
    let val = fmtValue(obs.value);
    if (val.length > 10) val = val.substring(0, 10);
    lines.push(`${name}: ${val}`);
    return;
  }

  if (isCoded(ciel, datatype)) {
    const answerDisplay = typeof obs.value === 'object'
      ? (obs.value?.display ?? String(obs.value))
      : String(obs.value);
    lines.push(`${name}: ${stripCiel(answerDisplay)}`);
    return;
  }

  let val = fmtValue(obs.value);
  if (!val) return;

  if (ciel && NUMERIC_UNITS[ciel] !== undefined) {
    const num = parseFloat(val);
    if (!isNaN(num)) {
      const units = NUMERIC_UNITS[ciel];
      val = units ? `${val} ${units}` : val;
    }
  }

  lines.push(`${name}: ${val}`);
}

// ── Main builder ───────────────────────────────────────────────────

export function buildCaseText(
  patient: PatientFull,
  encounters: Encounter[],
): string {
  const person = patient.person;
  const gender = (person.gender ?? 'U').charAt(0);
  const age = person.age != null ? person.age.toFixed(1) : '0.0';
  const birthdate = (person.birthdate ?? '').substring(0, 10) || '1900-01-01';

  let primaryId = 'Unknown';
  for (const id of patient.identifiers ?? []) {
    if (id.identifierType?.display?.toLowerCase().includes('openmrs')) {
      primaryId = id.identifier;
      break;
    }
  }
  if (primaryId === 'Unknown' && patient.identifiers?.length) {
    primaryId = patient.identifiers[0].identifier;
  }

  const lines: string[] = [
    `Age: ${age}`,
    `Gender: ${gender}`,
    `Birthdate: ${birthdate}`,
    `OpenMRS ID: ${primaryId}`,
  ];

  for (const attr of person.attributes ?? []) {
    const attrType = attr.attributeType?.display ?? '';
    let value = typeof attr.value === 'object'
      ? (attr.value?.display ?? String(attr.value))
      : String(attr.display ?? attr.value ?? '');
    if (value.includes(' = ')) value = value.split(' = ').slice(1).join(' = ');
    if (!value.trim()) continue;
    lines.push(`${attrType}: ${stripCiel(value.trim())}`);
  }

  for (const enc of encounters) {
    const encType = enc.encounterType?.display ?? 'Visit';
    let encDt = enc.encounterDatetime ?? '';
    if (encDt.length > 19) encDt = encDt.substring(0, 19);
    if (encDt && !encDt.includes('T')) encDt = encDt.substring(0, 10) + 'T00:00:00';
    const location = enc.location?.display ?? 'Unknown';

    lines.push(`Visit: ${encType}`);
    lines.push(`Date: ${encDt}`);
    lines.push(`Location: ${location}`);

    for (const obs of enc.obs ?? []) {
      renderObs(lines, obs);
    }
  }

  return lines.join('\n');
}

// ── Prompt wrapper ─────────────────────────────────────────────────

const CDS_PREAMBLE = `[CDS]\nProvide evidence-based clinical decision support for this patient.`;

export function buildModelPrompt(caseText: string): string {
  return (
    `<bos><start_of_turn>user\n` +
    `${CDS_PREAMBLE}\n` +
    `\n` +
    `${caseText}<end_of_turn>\n` +
    `<start_of_turn>model\n`
  );
}
