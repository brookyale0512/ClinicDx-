import { openmrsFetch, restBaseUrl } from '@openmrs/esm-framework';

// ── Types ──────────────────────────────────────────────────────────

export type ConceptDatatype =
  | 'Coded'
  | 'Numeric'
  | 'Text'
  | 'Date'
  | 'Datetime'
  | 'Boolean'
  | string;

export interface ConceptMapping {
  conceptReferenceTerm: {
    code: string;
    conceptSource: { display: string };
  };
}

export interface ObsConcept {
  uuid: string;
  display: string;
  datatype: { display: ConceptDatatype };
  conceptClass: { display: string };
  mappings: ConceptMapping[];
}

export interface Obs {
  uuid: string;
  display: string;
  concept: ObsConcept;
  value: string | number | boolean | { display: string; uuid?: string } | null;
  groupMembers: Obs[] | null;
}

export interface Encounter {
  uuid: string;
  encounterDatetime: string;
  encounterType: { display: string };
  location: { display: string } | null;
  obs: Obs[];
}

export interface PersonAttribute {
  attributeType: { display: string };
  display: string;
  value: string | number | boolean | { display: string } | null;
}

export interface PatientIdentifier {
  identifier: string;
  identifierType: { display: string };
}

export interface PatientFull {
  uuid: string;
  display: string;
  person: {
    uuid: string;
    gender: string;
    age: number;
    birthdate: string;
    preferredName: { givenName: string; familyName: string };
    attributes: PersonAttribute[];
  };
  identifiers: PatientIdentifier[];
}

// ── Fetch helpers ──────────────────────────────────────────────────

const PATIENT_REP =
  'custom:(uuid,display,' +
  'identifiers:(identifier,identifierType:(display)),' +
  'person:(uuid,gender,age,birthdate,' +
  'preferredName:(givenName,familyName),' +
  'attributes:(attributeType:(display),display,value)))';

const ENCOUNTER_REP =
  'custom:(uuid,encounterDatetime,' +
  'encounterType:(display),' +
  'location:(display),' +
  'obs:(uuid,display,' +
  'concept:(uuid,display,conceptClass:(display),datatype:(display),' +
  'mappings:(conceptReferenceTerm:(code,conceptSource:(display)))),' +
  'value,' +
  'groupMembers:(uuid,display,' +
  'concept:(uuid,display,datatype:(display),' +
  'mappings:(conceptReferenceTerm:(code,conceptSource:(display)))),' +
  'value)))';

export async function fetchPatient(patientUuid: string): Promise<PatientFull> {
  const resp = await openmrsFetch(
    `${restBaseUrl}/patient/${patientUuid}?v=${PATIENT_REP}`,
  );
  return resp.data as PatientFull;
}

export async function fetchEncounters(
  patientUuid: string,
  limit = 50,
): Promise<Encounter[]> {
  const resp = await openmrsFetch(
    `${restBaseUrl}/encounter?patient=${patientUuid}&v=${ENCOUNTER_REP}&limit=${limit}&order=asc`,
  );
  // API already returns results in ascending order — no client-side sort needed (Q-6)
  return (resp.data?.results ?? []) as Encounter[];
}
