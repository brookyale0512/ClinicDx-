import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  InlineLoading,
  Tile,
  Tag,
} from '@carbon/react';
import {
  WatsonHealthAiResultsHigh,
  WarningFilled,
  Stethoscope,
  Report,
  TaskComplete,
  WarningAlt,
  ListChecked,
  DataBase,
  Medication,
} from '@carbon/react/icons';
import { usePatient, useConfig, showSnackbar } from '@openmrs/esm-framework';
import {
  fetchPatient,
  fetchEncounters,
  buildCaseText,
  buildModelPrompt,
  generateCdsStreaming,
} from './case-builder';
import type { KbQueryResult, StreamEvent } from './case-builder';
import styles from './cds-workspace.scss';

interface CdsWorkspaceProps {
  patientUuid: string;
  closeWorkspace: (options?: { ignoreChanges?: boolean }) => void;
  promptBeforeClosing: (hasUnsavedChanges: () => boolean) => void;
}

/** @internal */
export interface ParsedSection {
  title: string;
  icon: string;
  content: string;
}

/** @internal — exported for unit testing */
export function cleanResponse(raw: string): string {
  let text = raw;
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  text = text.replace(/<kb_query>[\s\S]*?<\/kb_query>/gi, '');
  text = text.replace(/<think>[\s\S]*$/gi, '');
  text = text.replace(/<\/think>/gi, '');
  text = text.replace(/DECISION:\s*\w+/gi, '');
  return text.trim();
}

/** @internal — exported for unit testing */
export function parseModelResponse(raw: string): ParsedSection[] {
  const cleaned = cleanResponse(raw);
  const sections: ParsedSection[] = [];
  const sectionPattern = /^##\s+(.+)$/gm;
  const matches = [...cleaned.matchAll(sectionPattern)];

  if (matches.length === 0) {
    if (cleaned.length > 20) {
      return [{ title: 'Clinical Insight', icon: 'report', content: cleaned }];
    }
    return [];
  }

  const seen = new Set<string>();
  for (let i = 0; i < matches.length; i++) {
    const title = matches[i][1].trim();
    const key = title.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);

    const start = matches[i].index! + matches[i][0].length;
    const end = i + 1 < matches.length ? matches[i + 1].index! : cleaned.length;
    let content = cleaned.slice(start, end).trim();
    content = content.replace(/^## .+$/gm, '').trim();
    if (!content) continue;

    let icon = 'report';
    if (key.includes('clinical assessment')) icon = 'stethoscope';
    else if (key.includes('evidence') || key.includes('consideration')) icon = 'database';
    else if (key.includes('suggested action') || key.includes('treatment')) icon = 'treatment';
    else if (key.includes('safety alert') || key.includes('safety')) icon = 'shield';
    else if (key.includes('key point')) icon = 'list';

    sections.push({ title, icon, content });
  }
  return sections;
}

function renderMarkdown(content: string): React.ReactNode {
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];
  let key = 0;

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={key++} className={styles.mdList}>
          {listItems.map((item, i) => (
            <li key={i}>{renderInline(item)}</li>
          ))}
        </ul>,
      );
      listItems = [];
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) { flushList(); continue; }
    if (/^###\s+/.test(trimmed)) {
      flushList();
      elements.push(<h6 key={key++} className={styles.mdH3}>{trimmed.replace(/^###\s+/, '')}</h6>);
    } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      listItems.push(trimmed.slice(2));
    } else if (/^\d+\.\s/.test(trimmed)) {
      listItems.push(trimmed.replace(/^\d+\.\s/, ''));
    } else {
      flushList();
      elements.push(<p key={key++} className={styles.mdParagraph}>{renderInline(trimmed)}</p>);
    }
  }
  flushList();
  return <>{elements}</>;
}

function renderInline(text: string): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;
  while (remaining.length > 0) {
    const m = remaining.match(/\*\*(.+?)\*\*/);
    if (m && m.index !== undefined) {
      if (m.index > 0) parts.push(<span key={key++}>{remaining.slice(0, m.index)}</span>);
      parts.push(<strong key={key++}>{m[1]}</strong>);
      remaining = remaining.slice(m.index + m[0].length);
    } else {
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }
  }
  return <>{parts}</>;
}

const SectionIcon: React.FC<{ icon: string }> = ({ icon }) => {
  const props = { size: 20 };
  switch (icon) {
    case 'stethoscope': return <Stethoscope {...props} />;
    case 'database': return <DataBase {...props} />;
    case 'treatment': return <Medication {...props} />;
    case 'shield': return <WarningAlt {...props} />;
    case 'list': return <ListChecked {...props} />;
    default: return <Report {...props} />;
  }
};

const CdsWorkspace: React.FC<CdsWorkspaceProps> = ({
  patientUuid,
  closeWorkspace,
  promptBeforeClosing,
}) => {
  const { t } = useTranslation();
  const config = useConfig<{ middlewareUrl: string }>();
  const { patient, isLoading: isLoadingPatient } = usePatient(patientUuid);

  const [phase, setPhase] = useState<'idle' | 'building' | 'streaming' | 'done' | 'error'>('idle');
  const [parsedSections, setParsedSections] = useState<ParsedSection[]>([]);
  const [encounterCount, setEncounterCount] = useState(0);
  const [obsCount, setObsCount] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [streamedRaw, setStreamedRaw] = useState('');
  const [kbQueries, setKbQueries] = useState<KbQueryResult[]>([]);
  const [turnCount, setTurnCount] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const parseTimerRef = useRef<number | null>(null);

  useEffect(() => {
    // C-6: block close while actively building or streaming
    promptBeforeClosing(() => phase === 'streaming' || phase === 'building');
  }, [promptBeforeClosing, phase]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (parseTimerRef.current) clearTimeout(parseTimerRef.current);
    };
  }, []);

  const handleAnalyze = useCallback(async () => {
    setPhase('building');
    setErrorMsg(null);
    setParsedSections([]);
    setStreamedRaw('');
    setKbQueries([]);
    setTurnCount(0);

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const [patientData, encounters] = await Promise.all([
        fetchPatient(patientUuid),
        fetchEncounters(patientUuid),
      ]);

      const caseText = buildCaseText(patientData, encounters);
      const prompt = buildModelPrompt(caseText);
      const totalObs = encounters.reduce((sum, enc) => sum + (enc.obs?.length ?? 0), 0);

      setEncounterCount(encounters.length);
      setObsCount(totalObs);
      setPhase('streaming');

      let accumulated = '';
      let parseDirty = false;

      const scheduleParse = () => {
        if (parseTimerRef.current) return;
        parseDirty = true;
        parseTimerRef.current = window.setTimeout(() => {
          parseTimerRef.current = null;
          if (parseDirty) {
            setParsedSections(parseModelResponse(accumulated));
            parseDirty = false;
          }
        }, 150);
      };

      await generateCdsStreaming(
        config.middlewareUrl,
        prompt,
        (event: StreamEvent) => {
          switch (event.type) {
            case 'turn_start':
              setTurnCount(event.turn ?? 0);
              break;
            case 'token':
              accumulated += event.text ?? '';
              setStreamedRaw(accumulated);
              scheduleParse();
              break;
            case 'kb_result':
              setKbQueries((prev) => [
                ...prev,
                { query: event.query ?? '', score: event.score ?? 0, source: event.source ?? 'none' },
              ]);
              break;
            case 'done':
              setTurnCount(event.turns ?? 0);
              break;
            case 'error':
              setErrorMsg(event.message ?? 'Stream error');
              setPhase('error');
              // C-2: abort stream on error event
              controller.abort();
              break;
          }
        },
        controller.signal,
      );

      if (controller.signal.aborted) return;
      if (parseTimerRef.current) { clearTimeout(parseTimerRef.current); parseTimerRef.current = null; }
      setParsedSections(parseModelResponse(accumulated));
      setPhase('done');

      showSnackbar({
        title: t('analysisComplete', 'Analysis Complete'),
        kind: 'success',
        subtitle: 'Analysis complete',
      });
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error ? err.message : 'Unknown error';
      setErrorMsg(message);
      setPhase('error');
      showSnackbar({ title: t('error', 'Error'), kind: 'error', subtitle: message });
    }
  }, [patientUuid, config.middlewareUrl, t]);

  if (isLoadingPatient) {
    return (
      <div className={styles.centerBox}>
        <InlineLoading description={t('loadingPatient', 'Loading patient data...')} />
      </div>
    );
  }

  return (
    <div className={styles.workspace}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerTop}>
          <WatsonHealthAiResultsHigh size={24} />
          <h4>{t('clinicDxCds', 'ClinicDx CDS')}</h4>
          <Tag type="blue" size="sm">{t('aiPowered', 'AI-Powered')}</Tag>
        </div>
        <p className={styles.headerSub}>{t('clinicDxSubtitle', 'AI Clinical Decision Support System')}</p>
      </div>

      {/* Screen-reader live region for phase status (A-2) */}
      <div role="status" aria-live="polite" className={styles.srOnly}>
        {phase === 'building' && t('buildingCaseTitle', 'Preparing patient case...')}
        {phase === 'streaming' && t('modelWorking', 'Querying knowledge base and analyzing')}
        {phase === 'done' && t('analysisComplete', 'Analysis Complete')}
        {phase === 'error' && errorMsg}
      </div>

      <div className={styles.body}>
        {/* Patient Card */}
        <Tile className={styles.patientCard}>
          <h5>{t('patientContext', 'Patient Context')}</h5>
          {patient && (
            <div className={styles.patientGrid}>
              <div><span className={styles.label}>{t('name', 'Name')}</span><span className={styles.value}>{patient.name?.[0]?.given?.join(' ')} {patient.name?.[0]?.family}</span></div>
              <div><span className={styles.label}>{t('gender', 'Gender')}</span><span className={styles.value}>{patient.gender}</span></div>
              <div><span className={styles.label}>{t('birthDate', 'Birth Date')}</span><span className={styles.value}>{patient.birthDate}</span></div>
            </div>
          )}
        </Tile>

        {/* Action Button */}
        <Button
          className={styles.actionBtn}
          kind="primary"
          size="lg"
          renderIcon={phase === 'idle' || phase === 'done' || phase === 'error' ? WatsonHealthAiResultsHigh : undefined}
          onClick={handleAnalyze}
          disabled={phase === 'building' || phase === 'streaming'}
        >
          {phase === 'building'
            ? t('buildingCase', 'Building case...')
            : phase === 'streaming'
              ? t('analyzing', 'Analyzing with AI...')
              : t('getAiInsight', 'Get AI Insight')}
        </Button>

        {/* Streaming / Building indicator */}
        {phase === 'building' && (
          <Tile className={styles.analyzingCard}>
            <InlineLoading description="" />
            <div>
              <h5>{t('buildingCaseTitle', 'Preparing patient case...')}</h5>
              <p>{t('buildingCaseDetail', 'Collecting encounters and observations from OpenMRS.')}</p>
            </div>
          </Tile>
        )}

        {phase === 'streaming' && parsedSections.length === 0 && (
          <Tile className={styles.analyzingCard}>
            <InlineLoading description="" />
            <div>
              <h5>{t('modelWorking', 'Querying knowledge base and analyzing')}</h5>
              <p>{t('modelWorkingDetail', 'Streaming response — content will appear below.')}</p>
            </div>
            <div className={styles.tagRow}>
              <Tag type="teal" size="sm">{encounterCount} encounters</Tag>
              <Tag type="teal" size="sm">{obsCount} observations</Tag>
              {turnCount > 0 && <Tag type="purple" size="sm">Turn {turnCount}</Tag>}
            </div>
          </Tile>
        )}

        {/* Error */}
        {phase === 'error' && errorMsg && (
          <Tile className={styles.errorCard}>
            <WarningFilled size={20} />
            <p>{errorMsg}</p>
          </Tile>
        )}

        {/* Parsed section cards — shown progressively during streaming AND after done */}
        {(phase === 'streaming' || phase === 'done') && parsedSections.length > 0 && (
          <div className={styles.results}>
            {phase === 'streaming' && (
              <div className={`${styles.tagRow} ${styles.tagRowStreaming}`}>
                <InlineLoading description="" />
                <Tag type="purple" size="sm">Turn {turnCount || 1}</Tag>
                <Tag type="teal" size="sm">{encounterCount} encounters</Tag>
                <Tag type="teal" size="sm">{obsCount} obs</Tag>
              </div>
            )}

            {parsedSections.map((sec) => (
              <Tile key={sec.title} className={`${styles.sectionCard} ${styles['icon_' + sec.icon] || ''}`}>
                <div className={styles.sectionHead}>
                  <SectionIcon icon={sec.icon} />
                  <h5>{sec.title}</h5>
                </div>
                <div className={styles.sectionBody}>{renderMarkdown(sec.content)}</div>
              </Tile>
            ))}
          </div>
        )}

        {/* KB Evidence Sources — shown whenever KB results exist in done/streaming phase */}
        {(phase === 'streaming' || phase === 'done') && kbQueries.length > 0 && (
          <Tile className={styles.kbFeed}>
            <h6><DataBase size={16} /> {t('evidenceSources', 'Evidence Sources')}</h6>
            {kbQueries.map((kb, i) => (
              <div key={i} className={styles.kbEntry}>
                <Tag type="teal" size="sm">{`Q${i + 1}`}</Tag>
                <span className={styles.kbQuery}>{kb.query}</span>
                {/* A-6: text label alongside color-coded score */}
                <Tag type={kb.score > 30 ? 'green' : 'gray'} size="sm">
                  {kb.score > 30 ? t('highConfidence', 'High') : t('lowConfidence', 'Low')} — {kb.source} ({Math.round(kb.score)})
                </Tag>
              </div>
            ))}
          </Tile>
        )}

        {/* Metadata + Close button — always shown when analysis is complete (R-2) */}
        {phase === 'done' && (
          <div className={styles.metaFooter}>
            <Tag type="gray" size="sm">{turnCount} turns</Tag>
            <Tag type="gray" size="sm">{kbQueries.length} KB queries</Tag>
            <Tag type="gray" size="sm">{encounterCount} encounters</Tag>
            <Tag type="gray" size="sm">{obsCount} obs</Tag>
            <Button
              kind="ghost"
              size="sm"
              onClick={() => closeWorkspace({ ignoreChanges: true })}
            >
              {t('closeWorkspace', 'Close')}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default CdsWorkspace;
