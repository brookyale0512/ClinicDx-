import { getAsyncLifecycle, defineConfigSchema } from '@openmrs/esm-framework';

const moduleName = '@clinicdx/esm-clinicdx-app';

const options = {
  featureName: 'clinicdx',
  moduleName,
};

export const configSchema = {
  middlewareUrl: {
    _type: 'String',
    _default: '/clinicdx-api',
    _description:
      'Base URL for the ClinicDx middleware API. ' +
      'In Docker deployments this is /clinicdx-api (proxied by nginx). ' +
      'For local development set to http://localhost:8080.',
  },
};

export const importTranslation = require.context('../translations', false, /.json$/, 'lazy');

export function startupApp() {
  defineConfigSchema(moduleName, configSchema);
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
