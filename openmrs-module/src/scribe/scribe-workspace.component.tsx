import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  Tile,
  Tag,
  InlineLoading,
  Checkbox,
  Dropdown,
} from '@carbon/react';
import { Microphone, MicrophoneFilled, Checkmark, TrashCan, Add } from '@carbon/react/icons';
import { usePatient, useVisit, useConfig, openmrsFetch, restBaseUrl, showSnackbar } from '@openmrs/esm-framework';
import styles from './scribe-workspace.scss';

interface ScribeWorkspaceProps {
  patientUuid: string;
  closeWorkspace: (options?: { ignoreChanges?: boolean }) => void;
  promptBeforeClosing: (hasUnsavedChanges: () => boolean) => void;
}

interface EncounterOption {
  uuid: string;
  display: string;
  encounterType: string;
  datetime: string;
}

interface ExtractedItem {
  id: string;
  label: string;
  value: string;
  human_readable: string;
  checked: boolean;
  fhir_type?: string;
  fhir_payload?: Record<string, unknown>;
}

const ENCOUNTER_REP =
  'custom:(uuid,encounterDatetime,encounterType:(uuid,display),visit:(uuid))';

const SKIP_PREFIXES = [
  'i need', 'i will', 'let me', 'the patient', 'observations',
  'concept', 'extract', 'here', 'based on', 'from the',
  'audio', 'manifest', 'note', 'output', 'result',
];

/** Max recording duration before auto-stop (5 minutes) */
const MAX_RECORDING_MS = 300_000;

export function parseItems(raw: string): ExtractedItem[] {
  let text = raw.replace(/<think>[\s\S]*?<\/think>/g, '');
  text = text.replace(/<think>[\s\S]*/g, '');

  const items: ExtractedItem[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim().replace(/^[-*]\s*/, '');
    if (!trimmed || !trimmed.includes(':')) continue;
    const [label, ...rest] = trimmed.split(':');
    const value = rest.join(':').trim();
    const key = label.trim().toLowerCase();
    if (!key || !value) continue;
    if (SKIP_PREFIXES.some((p) => key.startsWith(p))) continue;
    if (key.length > 50 || key.split(/[_ ]/).length > 5) continue;
    const cleanLabel = label.trim().replace(/_/g, ' ');
    items.push({
      id: `item-${items.length}`,
      label: label.trim(),
      value,
      human_readable: `${cleanLabel}: ${value}`,
      checked: true,
    });
  }
  return items;
}

export function handleExtractedData(data: {
  items?: Array<{ label: string; value: string; human_readable?: string; fhir_type?: string; fhir_payload?: Record<string, unknown> }>;
  raw_model_output?: string;
  raw_output?: string;
}): ExtractedItem[] {
  if (data.items && Array.isArray(data.items)) {
    return data.items.map((it, i) => ({
      id: `item-${i}`,
      label: it.label,
      value: it.value,
      human_readable: it.human_readable || `${it.label}: ${it.value}`,
      checked: true,
      fhir_type: it.fhir_type,
      fhir_payload: it.fhir_payload,
    }));
  }
  return parseItems(data.raw_model_output || data.raw_output || '');
}

function formatEncounterDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      day: 'numeric', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

const ScribeWorkspace: React.FC<ScribeWorkspaceProps> = ({
  patientUuid,
  closeWorkspace,
  promptBeforeClosing,
}) => {
  const { t } = useTranslation();
  const config = useConfig<{ middlewareUrl: string; vitalsEncounterTypeUuid: string }>();
  const { patient, isLoading: isLoadingPatient } = usePatient(patientUuid);
  const { activeVisit } = useVisit(patientUuid);

  const [phase, setPhase] = useState<'idle' | 'recording' | 'processing' | 'review' | 'confirmed'>('idle');
  const [recordingTime, setRecordingTime] = useState(0);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [items, setItems] = useState<ExtractedItem[]>([]);
  const [rawOutput, setRawOutput] = useState('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [encounters, setEncounters] = useState<EncounterOption[]>([]);
  const [selectedEncounter, setSelectedEncounter] = useState<EncounterOption | null>(null);
  const [manifestString, setManifestString] = useState('');
  const [lookup, setLookup] = useState<Record<string, unknown>>({});
  const [isLoadingEncounters, setIsLoadingEncounters] = useState(false);
  const [isLoadingManifest, setIsLoadingManifest] = useState(false);
  const [isCreatingEncounter, setIsCreatingEncounter] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const stopRecordingRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    promptBeforeClosing(() => phase === 'review');
  }, [promptBeforeClosing, phase]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      mediaRecorderRef.current?.stop();
    };
  }, []);

  // S-4: auto-stop recording after MAX_RECORDING_MS — via effect to avoid circular dep
  useEffect(() => {
    if (phase !== 'recording') return;
    const timer = window.setTimeout(() => {
      stopRecordingRef.current?.();
    }, MAX_RECORDING_MS);
    return () => clearTimeout(timer);
  }, [phase]);

  // Fetch encounters for the patient on mount
  useEffect(() => {
    if (!patientUuid) return;
    let cancelled = false;
    setIsLoadingEncounters(true);

    openmrsFetch(
      `${restBaseUrl}/encounter?patient=${patientUuid}&v=${ENCOUNTER_REP}&limit=30&order=desc`,
    ).then((resp) => {
      if (cancelled) return;
      const results = (resp.data?.results ?? []) as Array<{
        uuid: string;
        encounterType?: { display?: string };
        encounterDatetime: string;
      }>;
      const opts: EncounterOption[] = results.map((enc) => ({
        uuid: enc.uuid,
        display: `${enc.encounterType?.display ?? t('encounter', 'Encounter')} — ${formatEncounterDate(enc.encounterDatetime)}`,
        encounterType: enc.encounterType?.display ?? '',
        datetime: enc.encounterDatetime,
      }));
      setEncounters(opts);
      if (opts.length > 0) setSelectedEncounter(opts[0]);
    }).catch((err: Error) => {
      if (!cancelled) setErrorMsg(`Failed to load encounters: ${err.message}`);
    }).finally(() => {
      if (!cancelled) setIsLoadingEncounters(false);
    });

    return () => { cancelled = true; };
  }, [patientUuid, t]);

  // Fetch manifest when selected encounter changes
  useEffect(() => {
    if (!selectedEncounter) {
      setManifestString('');
      setLookup({});
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    setIsLoadingManifest(true);

    fetch(`${config.middlewareUrl}/scribe/manifest?encounter_uuid=${selectedEncounter.uuid}`, {
      signal: controller.signal,
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
    })
      .then((r) => {
        if (!r.ok) throw new Error(`Manifest error (${r.status})`);
        return r.json() as Promise<{ manifest_string?: string; lookup?: Record<string, unknown> }>;
      })
      .then((data) => {
        if (cancelled) return;
        setManifestString(data.manifest_string || '');
        setLookup(data.lookup || {});
      })
      .catch(() => {
        if (!cancelled) {
          setManifestString('');
          setLookup({});
        }
      })
      .finally(() => {
        clearTimeout(timeout);
        if (!cancelled) setIsLoadingManifest(false);
      });

    return () => { cancelled = true; controller.abort(); clearTimeout(timeout); };
  }, [selectedEncounter, config.middlewareUrl]);

  const handleNewEncounter = useCallback(async () => {
    if (!activeVisit) {
      setErrorMsg(t('noActiveVisit', 'No active visit. Start a visit for this patient first.'));
      return;
    }
    setIsCreatingEncounter(true);
    setErrorMsg(null);
    try {
      const resp = await openmrsFetch(`${restBaseUrl}/encounter`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: {
          patient: patientUuid,
          // R-1: use config instead of hardcoded UUID
          encounterType: config.vitalsEncounterTypeUuid,
          visit: activeVisit.uuid,
        },
      });
      const enc = resp.data as {
        uuid: string;
        encounterType?: { display?: string };
        encounterDatetime: string;
      };
      const opt: EncounterOption = {
        uuid: enc.uuid,
        display: `${enc.encounterType?.display ?? t('vitals', 'Vitals')} — ${formatEncounterDate(enc.encounterDatetime)}`,
        encounterType: enc.encounterType?.display ?? 'Vitals',
        datetime: enc.encounterDatetime,
      };
      setEncounters((prev) => [opt, ...prev]);
      setSelectedEncounter(opt);
      showSnackbar({ title: t('encounterCreated', 'Encounter Created'), kind: 'success', subtitle: opt.display });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setErrorMsg(`Failed to create encounter: ${message}`);
    } finally {
      setIsCreatingEncounter(false);
    }
  }, [patientUuid, activeVisit, config.vitalsEncounterTypeUuid, t]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  const startRecording = useCallback(async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onerror = () => {
        setErrorMsg(t('recordingError', 'Recording error. Check microphone permissions.'));
      };

      // C-1: assign onstop BEFORE calling start/stop to avoid race condition
      recorder.onstop = () => { /* handled in stopRecording promise */ };

      recorder.start(250);
      mediaRecorderRef.current = recorder;

      setPhase('recording');
      setTranscription(null);
      setItems([]);
      startTimeRef.current = Date.now();
      timerRef.current = window.setInterval(() => {
        setRecordingTime((Date.now() - startTimeRef.current) / 1000);
      }, 100);
    } catch (e: unknown) {
      if (e instanceof Error && e.name === 'NotAllowedError') {
        setErrorMsg(t('micDenied', 'Microphone permission denied. Allow microphone access in browser settings.'));
      } else {
        const message = e instanceof Error ? e.message : 'Unknown error';
        setErrorMsg(t('micError', 'Could not start recording: {{message}}', { message }));
      }
    }
  }, [t]);

  const stopRecording = useCallback(async () => {
    if (phase !== 'recording' || !mediaRecorderRef.current) return;

    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setRecordingTime(0);

    // C-4: guard — selectedEncounter required
    if (!selectedEncounter) {
      setErrorMsg(t('selectEncounterFirst', 'Select an encounter first'));
      setPhase('idle');
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((t) => t.stop());
      return;
    }

    const audioBlob = await new Promise<Blob>((resolve) => {
      // C-1: assign onstop BEFORE stop() to avoid race condition
      mediaRecorderRef.current!.onstop = () => {
        resolve(new Blob(audioChunksRef.current, { type: 'audio/webm' }));
      };
      mediaRecorderRef.current!.stop();
      mediaRecorderRef.current!.stream.getTracks().forEach((track) => track.stop());
    });

    if (audioBlob.size < 1000) {
      setPhase('idle');
      setErrorMsg(t('recordingTooShort', 'Recording too short. Speak for at least a second.'));
      return;
    }

    setTranscription(t('processingAudioHint', '[audio recorded — processing directly...]'));
    setPhase('processing');

    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('encounter_uuid', selectedEncounter.uuid);
      formData.append('patient_uuid', patientUuid);
      formData.append('manifest_string', manifestString);
      formData.append('lookup', JSON.stringify(lookup));

      const res = await fetch(`${config.middlewareUrl}/scribe/process_audio`, {
        method: 'POST',
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as { detail?: string };
        throw new Error(body.detail || `Server error (${res.status})`);
      }

      const data = await res.json() as {
        raw_model_output?: string;
        raw_output?: string;
        transcription?: string;
        items?: Array<{ label: string; value: string; human_readable?: string; fhir_type?: string; fhir_payload?: Record<string, unknown> }>;
      };
      setRawOutput(data.raw_model_output || data.raw_output || '');
      setTranscription(data.transcription || t('directAudio', '[direct audio — no transcript]'));
      setItems(handleExtractedData(data));
      setPhase('review');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : t('processingFailed', 'Processing failed');
      setErrorMsg(message);
      setPhase('idle');
    }
  }, [phase, patientUuid, selectedEncounter, manifestString, lookup, config.middlewareUrl, t]);

  // Keep stopRecordingRef in sync with the latest stopRecording (for max-duration effect)
  useEffect(() => {
    stopRecordingRef.current = stopRecording;
  }, [stopRecording]);

  const toggleItem = useCallback((id: string) => {
    setItems((prev) => prev.map((it) => it.id === id ? { ...it, checked: !it.checked } : it));
  }, []);

  const handleConfirm = useCallback(async () => {
    // C-4: programmatic guard
    if (!selectedEncounter) {
      setErrorMsg(t('selectEncounterFirst', 'Select an encounter first'));
      return;
    }
    const checked = items.filter((it) => it.checked);
    if (!checked.length) return;

    setPhase('processing');
    try {
      const res = await fetch(`${config.middlewareUrl}/scribe/confirm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
        },
        body: JSON.stringify({
          encounter_uuid: selectedEncounter.uuid,
          patient_uuid: patientUuid,
          items: checked.map((it) => ({
            id: it.id,
            label: it.label,
            value: it.value,
            fhir_type: it.fhir_type ?? 'Observation',
            fhir_payload: it.fhir_payload ?? {},
          })),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as { detail?: string };
        throw new Error(body.detail || `Confirm failed (${res.status})`);
      }
      const result = await res.json() as { posted: number; failed: number };
      setPhase('confirmed');
      showSnackbar({
        title: t('observationsSaved', 'Observations Saved to OpenMRS'),
        kind: 'success',
        subtitle: `${result.posted} posted, ${result.failed} failed`,
      });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : t('confirmFailed', 'Confirm failed');
      setErrorMsg(message);
      setPhase('review');
    }
  }, [items, selectedEncounter, patientUuid, config.middlewareUrl, t]);

  const handleReset = useCallback(() => {
    setPhase('idle');
    setTranscription(null);
    setItems([]);
    setRawOutput('');
    setErrorMsg(null);
  }, []);

  if (isLoadingPatient) {
    return (
      <div className={styles.centerBox}>
        <InlineLoading description={t('loadingPatient', 'Loading patient data...')} />
      </div>
    );
  }

  return (
    <div className={styles.workspace}>
      <div className={styles.header}>
        <div className={styles.headerTop}>
          <Microphone size={24} />
          <h4>{t('clinicDxScribe', 'ClinicDx Scribe')}</h4>
          <Tag type="blue" size="sm">{t('aiPowered', 'AI-Powered')}</Tag>
        </div>
        <p className={styles.headerSub}>{t('scribeSubtitle', 'Speak clinical phrases — AI extracts structured observations for OpenMRS')}</p>
      </div>

      {/* A-2: screen reader live region */}
      <div role="status" aria-live="polite" className={styles.srOnly}>
        {phase === 'recording' && t('recordingActive', 'Recording active')}
        {phase === 'processing' && t('processingAudio', 'Processing audio...')}
        {phase === 'review' && t('reviewReady', 'Observations ready for review')}
        {phase === 'confirmed' && t('observationsConfirmed', 'Observations Confirmed')}
      </div>

      <div className={styles.body}>
        {/* Patient */}
        <Tile className={styles.patientCard}>
          <h5>{t('patientContext', 'Patient Context')}</h5>
          {patient && (
            <div className={styles.patientGrid}>
              <div><span className={styles.label}>{t('name', 'Name')}</span><span className={styles.val}>{patient.name?.[0]?.given?.join(' ')} {patient.name?.[0]?.family}</span></div>
              <div><span className={styles.label}>{t('gender', 'Gender')}</span><span className={styles.val}>{patient.gender}</span></div>
              <div><span className={styles.label}>{t('birthDate', 'Birth Date')}</span><span className={styles.val}>{patient.birthDate}</span></div>
            </div>
          )}
        </Tile>

        {/* Encounter Selector */}
        <Tile className={styles.encounterCard}>
          <div className={styles.encounterRow}>
            {isLoadingEncounters ? (
              <InlineLoading description={t('loadingEncounters', 'Loading encounters...')} />
            ) : (
              <Dropdown
                id="encounter-select"
                titleText={t('encounter', 'Encounter')}
                label={encounters.length ? t('selectEncounter', 'Select an encounter') : t('noEncountersFound', 'No encounters found')}
                items={encounters}
                itemToString={(item: EncounterOption | null) => item?.display ?? ''}
                selectedItem={selectedEncounter}
                onChange={({ selectedItem }: { selectedItem: EncounterOption | null }) => {
                  setSelectedEncounter(selectedItem ?? null);
                }}
                className={styles.encounterDropdown}
                size="md"
              />
            )}
            <Button
              kind="ghost"
              size="md"
              renderIcon={Add}
              onClick={handleNewEncounter}
              disabled={isCreatingEncounter || !activeVisit}
              className={styles.encounterAddBtn}
              hasIconOnly
              iconDescription={t('newEncounter', 'New Encounter')}
              tooltipPosition="left"
            />
          </div>
          {isLoadingManifest && <InlineLoading description={t('preparingManifest', 'Preparing...')} className={styles.manifestLoading} />}
          {!activeVisit && encounters.length === 0 && (
            <p className={styles.encounterHint}>{t('noEncountersHint', 'No encounters found — start a visit and create an encounter first')}</p>
          )}
        </Tile>

        {/* Error */}
        {errorMsg && (
          <Tile className={styles.errorCard}><p>{errorMsg}</p></Tile>
        )}

        {/* Mic Button — click to start */}
        {(phase === 'idle' || phase === 'confirmed') && (
          <div className={styles.micSection}>
            <button
              className={styles.micBtn}
              onClick={startRecording}
              disabled={!selectedEncounter || isLoadingManifest}
              aria-label={t('holdToRecord', 'Hold to speak')}
            >
              <Microphone size={40} />
            </button>
            <p className={styles.micLabel}>
              {selectedEncounter ? t('tapToRecord', 'Tap to start recording') : t('selectEncounterFirst', 'Select an encounter first')}
            </p>
          </div>
        )}

        {/* Recording — click to stop */}
        {phase === 'recording' && (
          <div className={styles.micSection}>
            <button
              className={`${styles.micBtn} ${styles.micActive}`}
              onClick={stopRecording}
              aria-label={t('stopRecording', 'Stop recording')}
            >
              <MicrophoneFilled size={40} />
              <div className={styles.pulse} />
            </button>
            <p className={styles.micTime}>{formatTime(recordingTime)}</p>
            {transcription && <p className={styles.liveText}>{transcription}</p>}
            <div className={styles.waveform}>
              {Array.from({ length: 12 }).map((_, i) => (
                <div key={i} className={styles.bar} style={{ animationDelay: `${i * 0.05}s` }} />
              ))}
            </div>
            <Button kind="danger" size="lg" onClick={stopRecording} className={styles.stopBtn}>
              {t('stopAndProcess', 'Stop & Process')}
            </Button>
          </div>
        )}

        {/* Processing */}
        {phase === 'processing' && (
          <Tile className={styles.processingCard}>
            <InlineLoading description="" />
            <div>
              <h5>{t('processingAudio', 'Processing audio...')}</h5>
              <p>{t('extractingObservations', 'Extracting clinical observations directly from audio recording')}</p>
            </div>
          </Tile>
        )}

        {/* Review */}
        {phase === 'review' && transcription && (
          <div className={styles.reviewSection}>
            <Tile className={styles.transcriptionCard}>
              <Tag type="teal" size="sm">{t('audioProcessed', 'Audio Processed')}</Tag>
              <p className={styles.transcriptionText}>{transcription}</p>
            </Tile>

            {items.length > 0 ? (
              <>
                <div className={styles.itemsHeader}>
                  <h5>{t('extractedObservations', 'Extracted Observations')}</h5>
                  <Tag type="green" size="sm">{items.filter((i) => i.checked).length} selected</Tag>
                </div>

                {items.map((item) => (
                  <Tile
                    key={item.id}
                    className={`${styles.itemCard} ${item.checked ? styles.itemChecked : styles.itemUnchecked}`}
                    onClick={() => toggleItem(item.id)}
                  >
                    {/* A-3: use the human-readable label as labelText instead of empty string */}
                    <Checkbox
                      id={`chk-${item.id}`}
                      labelText={item.label.replace(/_/g, ' ')}
                      checked={item.checked}
                      onChange={() => toggleItem(item.id)}
                    />
                    <div className={styles.itemContent}>
                      <span className={styles.itemValue}>{item.value}</span>
                    </div>
                  </Tile>
                ))}

                <div className={styles.reviewActions}>
                  <Button
                    kind="primary"
                    size="lg"
                    renderIcon={Checkmark}
                    onClick={handleConfirm}
                    disabled={!selectedEncounter || !items.some((i) => i.checked)}
                    className={styles.confirmBtn}
                  >
                    {t('confirmCount', 'Confirm ({{count}})', { count: items.filter((i) => i.checked).length })}
                  </Button>
                  <Button kind="ghost" size="sm" renderIcon={TrashCan} onClick={handleReset}>
                    {t('discard', 'Discard')}
                  </Button>
                </div>
              </>
            ) : (
              <Tile className={styles.processingCard}>
                <p>{t('noObservationsExtracted', 'No observations extracted. Try speaking more clearly.')}</p>
                <Button kind="tertiary" size="sm" onClick={handleReset}>{t('tryAgain', 'Try Again')}</Button>
              </Tile>
            )}
          </div>
        )}

        {/* Confirmed */}
        {phase === 'confirmed' && (
          <div className={styles.confirmedSection}>
            <Tile className={styles.confirmedCard}>
              <Checkmark size={32} />
              <h5>{t('observationsConfirmed', 'Observations Confirmed')}</h5>
              <p>{t('itemsReadyForOpenmrs', '{{count}} items ready for OpenMRS', { count: items.filter((i) => i.checked).length })}</p>
            </Tile>
            <Button kind="tertiary" onClick={handleReset}>{t('recordAnother', 'Record Another Phrase')}</Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ScribeWorkspace;
