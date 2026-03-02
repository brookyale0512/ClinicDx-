export { fetchPatient, fetchEncounters } from './openmrs-api';
export { buildCaseText, buildModelPrompt } from './case-xml';
export { generateCds, generateCdsStreaming, checkCdsHealth } from './cds-api';
export type { CdsResult, KbQueryResult, StreamEvent } from './cds-api';
