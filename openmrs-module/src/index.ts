import { getAsyncLifecycle, defineConfigSchema, Type, registerFeatureFlag } from '@openmrs/esm-framework';

const moduleName = '@openmrs/esm-clinicdx-app';

const options = {
  featureName: 'clinicdx',
  moduleName,
};

export const configSchema = {
  middlewareUrl: {
    _type: Type.String,
    _default: '/clinicdx-api',
    _description: 'URL for the ClinicDx middleware API',
  },
  vitalsEncounterTypeUuid: {
    _type: Type.String,
    _default: '67a71486-1a54-468f-ac3e-7091a9a79584',
    _description: 'UUID of the Vitals encounter type used when creating new encounters from Scribe',
  },
};

export const importTranslation = require.context('../translations', false, /.json$/, 'lazy');

export function startupApp() {
  defineConfigSchema(moduleName, configSchema);
  registerFeatureFlag('clinicdx-ocr', 'ClinicDx OCR', 'Lab result OCR — under development');
  registerFeatureFlag('clinicdx-imaging', 'ClinicDx Imaging', 'DICOM imaging — under development');
}

// ── CDS ────────────────────────────────────────────────────────────

export const clinicDxCdsButton = getAsyncLifecycle(
  () => import('./cds/cds-action-button.component'),
  options,
);

export const clinicDxCdsWorkspace = getAsyncLifecycle(
  () => import('./cds/cds-workspace.component'),
  options,
);

// ── Scribe ─────────────────────────────────────────────────────────

export const clinicDxScribeButton = getAsyncLifecycle(
  () => import('./scribe/scribe-action-button.component'),
  options,
);

export const clinicDxScribeWorkspace = getAsyncLifecycle(
  () => import('./scribe/scribe-workspace.component'),
  options,
);

// ── OCR ────────────────────────────────────────────────────────────

export const clinicDxOcrButton = getAsyncLifecycle(
  () => import('./ocr/ocr-action-button.component'),
  options,
);

export const clinicDxOcrWorkspace = getAsyncLifecycle(
  () => import('./ocr/ocr-workspace.component'),
  options,
);

// ── Imaging ────────────────────────────────────────────────────────

export const clinicDxImagingButton = getAsyncLifecycle(
  () => import('./imaging/imaging-action-button.component'),
  options,
);

export const clinicDxImagingWorkspace = getAsyncLifecycle(
  () => import('./imaging/imaging-workspace.component'),
  options,
);
