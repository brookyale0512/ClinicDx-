import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  ComposedModal,
  InlineLoading,
  ModalBody,
  ModalFooter,
  ModalHeader,
  Tag,
  Tile,
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
  Idea,
} from '@carbon/react/icons';
import { usePatient, showSnackbar } from '@openmrs/esm-framework';
import {
  fetchPatient,
  fetchEncounters,
  buildCaseXml,
  buildModelPrompt,
  generateCdsStreaming,
} from './case-builder';
import type { KbHit, KbQueryResult, StreamEvent } from './case-builder';
import styles from './cds-workspace.scss';

interface CdsWorkspaceProps {
  patientUuid: string;
  closeWorkspace: (options?: { ignoreChanges?: boolean }) => void;
  promptBeforeClosing: (hasUnsavedChanges: () => boolean) => void;
}

export interface ParsedSection {
  title: string;
  icon: string;
  content: string;
}

export function cleanResponse(raw: string): string {
  let text = raw;
  // Strip complete think/thinking blocks
  text = text.replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '');
  text = text.replace(/<kb_query>[\s\S]*?<\/kb_query>/gi, '');
  // Strip any trailing unclosed think/thinking block
  text = text.replace(/<think(?:ing)?>[\s\S]*$/gi, '');
  // Strip any orphaned closing tags
  text = text.replace(/<\/think(?:ing)?>/gi, '');
  text = text.replace(/DECISION:\s*\w+/gi, '');
  return text.trim();
}

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

/**
 * Renders raw KB chunk text into a human-readable structured view.
 *
 * KB chunks contain:
 *   1. Clinical content (prose + "- item" bullets mixed inline)
 *   2. A metadata tail starting at "title:" which we strip entirely
 *
 * Steps:
 *   - Strip the metadata tail (title: … onwards)
 *   - Strip numeric citation markers like (74)
 *   - Split on ". - " or "\n- " to separate bullet items embedded in prose
 *   - Render bullets as <ul> and prose as <p>
 *   - Bold clinical dosing values (mg, mg/kg, g/dL, %, etc.)
 */
function renderKbContent(raw: string): React.ReactNode {
  // 1. Strip metadata tail — everything from "title:" onwards
  const metaStart = raw.search(/\btitle:\s/);
  const body = (metaStart > 0 ? raw.slice(0, metaStart) : raw).trim();

  // 2. Strip numeric citations (74), (117) etc.
  const cleaned = body.replace(/\s*\(\d{1,4}\)\s*/g, ' ').replace(/\s{2,}/g, ' ').trim();

  // 3. Tokenise: split on ". - " (bullet embedded after sentence) and on "- " at segment start
  //    Produce an array of { type: 'prose'|'bullet', text }
  type Seg = { type: 'prose' | 'bullet'; text: string };
  const segments: Seg[] = [];

  // First split on "\n- " or ". - " to extract embedded bullets
  const rawSegments = cleaned.split(/(?:\.\s+-\s+|(?:^|\n)\s*-\s+)/);

  rawSegments.forEach((seg, idx) => {
    const s = seg.trim();
    if (!s) return;
    // The first segment after a split is prose; subsequent ones are bullets
    // Exception: if cleaned text starts with "- " the very first is a bullet
    const isBullet = idx > 0 || cleaned.startsWith('- ');
    segments.push({ type: isBullet ? 'bullet' : 'prose', text: s });
  });

  // 4. Bold clinical values
  const boldClinical = (text: string): React.ReactNode => {
    const re = /(\d[\d.,/×³–-]*\s*(?:mg\/kg(?:\/day|\/dose|\/dose\/day)?|mg|g\/dL|g\/L|mmol\/L|μmol\/L|mL|IU\/(?:mL|kg)|IU|g|%|\/μL|mmHg|bpm|°C|weeks?|days?|months?|hours?|h|q\d+h|times?\s+daily|times?\s+a\s+day))/gi;
    const parts: React.ReactNode[] = [];
    let last = 0; let k = 0; let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) parts.push(<span key={k++}>{text.slice(last, m.index)}</span>);
      parts.push(<strong key={k++}>{m[1]}</strong>);
      last = m.index + m[0].length;
    }
    if (last < text.length) parts.push(<span key={k++}>{text.slice(last)}</span>);
    return <>{parts}</>;
  };

  // 5. Group consecutive bullets into a single <ul>
  const elements: React.ReactNode[] = [];
  let bulletBuf: string[] = [];
  let key = 0;

  const flushBullets = () => {
    if (bulletBuf.length === 0) return;
    elements.push(
      <ul key={key++} style={{ margin: '0.5rem 0 0.75rem 1.25rem', padding: 0, lineHeight: '1.7' }}>
        {bulletBuf.map((b, i) => (
          <li key={i} style={{ marginBottom: '0.35rem', color: 'var(--cds-text-secondary, #525252)' }}>
            {boldClinical(b)}
          </li>
        ))}
      </ul>,
    );
    bulletBuf = [];
  };

  for (const seg of segments) {
    if (seg.type === 'bullet') {
      bulletBuf.push(seg.text);
    } else {
      flushBullets();
      // Break prose into short paragraphs (~3 sentences)
      const sentences = seg.text.split(/(?<=[.!?])\s+(?=[A-Z])/);
      for (let i = 0; i < sentences.length; i += 3) {
        const para = sentences.slice(i, i + 3).join(' ').trim();
        if (para) {
          elements.push(
            <p key={key++} style={{ marginBottom: '0.75rem', lineHeight: '1.65', color: 'var(--cds-text-secondary, #525252)' }}>
              {boldClinical(para)}
            </p>,
          );
        }
      }
    }
  }
  flushBullets();

  return <>{elements}</>;
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

// Streams raw <think> content while the model reasons.
// When `hiding` becomes true the panel collapses (height + opacity) before
// the answer sections slide in — ensuring zero layout overlap.
const ThinkingPanel: React.FC<{ text: string; hiding: boolean }> = ({ text, hiding }) => {
  const preRef = useRef<HTMLPreElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Lock wrapper height before collapse so CSS transition has a start value
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    if (hiding) {
      // Pin current rendered height explicitly, then on next frame set to 0
      wrap.style.height = `${wrap.scrollHeight}px`;
      requestAnimationFrame(() => {
        wrap.style.height = '0';
      });
    } else {
      // Expanding: measure content and animate to it
      wrap.style.height = `${wrap.scrollHeight}px`;
      const onEnd = () => { wrap.style.height = ''; };
      wrap.addEventListener('transitionend', onEnd, { once: true });
    }
  }, [hiding]);

  // Scroll the workspace body down as thinking text grows so the cursor stays visible
  useEffect(() => {
    if (!hiding && wrapRef.current) {
      wrapRef.current.scrollIntoView({ block: 'end', behavior: 'smooth' });
    }
  }, [text, hiding]);

  return (
    <div
      ref={wrapRef}
      className={`${styles.thinkWrap}${hiding ? ` ${styles.thinkWrapCollapsing}` : ''}`}
    >
      <Tile className={styles.thinkStream}>
        <div className={styles.thinkHeader}>
          {!hiding && <InlineLoading />}
          <Idea size={14} />
          <span className={styles.thinkLabel}>
            {hiding ? 'Reasoning complete' : 'Reasoning...'}
          </span>
        </div>
        <pre ref={preRef} className={styles.thinkPre}>
          {text}
          {!hiding && <span className={styles.thinkCursor} aria-hidden="true" />}
        </pre>
      </Tile>
    </div>
  );
};

const CdsWorkspace: React.FC<CdsWorkspaceProps> = ({
  patientUuid,
  closeWorkspace,
  promptBeforeClosing,
}) => {
  const { t } = useTranslation();
  const { patient, isLoading: isLoadingPatient } = usePatient(patientUuid);

  const [phase, setPhase] = useState<'idle' | 'building' | 'streaming' | 'done' | 'error'>('idle');
  const [parsedSections, setParsedSections] = useState<ParsedSection[]>([]);
  const [encounterCount, setEncounterCount] = useState(0);
  const [obsCount, setObsCount] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [streamedRaw, setStreamedRaw] = useState('');
  const [kbHits, setKbHits] = useState<KbHit[]>([]);
  const [turnCount, setTurnCount] = useState(0);
  const [selectedHit, setSelectedHit] = useState<KbHit | null>(null);
  const [showAllHits, setShowAllHits] = useState(false);
  // Thinking panel state
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingText, setThinkingText] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  const parseTimerRef = useRef<number | null>(null);
  const streamErrorRef = useRef<boolean>(false);

  useEffect(() => {
    promptBeforeClosing(() => false);
  }, [promptBeforeClosing]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (parseTimerRef.current) clearTimeout(parseTimerRef.current);
    };
  }, []);

  // Hide the thinking panel as soon as at least one section is visible —
  // this avoids the flash that happens when isThinking flips off before
  // parsedSections are rendered.
  useEffect(() => {
    if (parsedSections.length > 0 && isThinking) {
      setIsThinking(false);
    }
  }, [parsedSections.length, isThinking]);

  const handleAnalyze = useCallback(async () => {
    setPhase('building');
    setErrorMsg(null);
    setParsedSections([]);
    setStreamedRaw('');
    setKbHits([]);
    setTurnCount(0);
    setShowAllHits(false);
    setSelectedHit(null);
    setIsThinking(false);
    setThinkingText('');

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    streamErrorRef.current = false;

    try {
      const [patientData, encounters] = await Promise.all([
        fetchPatient(patientUuid),
        fetchEncounters(patientUuid),
      ]);

      const caseText = buildCaseXml(patientData, encounters);
      const prompt = buildModelPrompt(caseText);
      const totalObs = encounters.reduce((sum, enc) => sum + (enc.obs?.length ?? 0), 0);

      setEncounterCount(encounters.length);
      setObsCount(totalObs);
      setPhase('streaming');

      // ── Accumulated raw text (kept for final parse on stream end) ──
      let accumulated = '';

      // ── Live section streaming ──
      // Instead of batch-parsing on a debounce timer, we detect ## headings
      // as they arrive and stream content character-by-character into the
      // active (last) section — giving the same live-token effect as the
      // thinking panel, one card at a time in order.
      //
      // liveSections mirrors parsedSections but is mutated in place.
      // We call setParsedSections with a shallow copy to trigger a React render.
      const liveSections: ParsedSection[] = [];

      // Buffer for the current line being assembled (to detect ## headings)
      let lineBuf = '';

      const getSectionIcon = (title: string): string => {
        const key = title.toLowerCase();
        if (key.includes('clinical assessment')) return 'stethoscope';
        if (key.includes('evidence') || key.includes('consideration')) return 'database';
        if (key.includes('suggested action') || key.includes('treatment')) return 'treatment';
        if (key.includes('safety alert') || key.includes('safety')) return 'shield';
        if (key.includes('key point')) return 'list';
        return 'report';
      };

      // Flush a single answer character into the live sections state machine.
      const pushAnswerChar = (ch: string) => {
        lineBuf += ch;

        if (ch === '\n') {
          const trimmed = lineBuf.trimEnd();
          const headingMatch = trimmed.match(/^##\s+(.+)$/);
          if (headingMatch) {
            const title = headingMatch[1].trim();
            // Only create a new section if title not already present
            if (!liveSections.find(s => s.title.toLowerCase() === title.toLowerCase())) {
              liveSections.push({ title, icon: getSectionIcon(title), content: '' });
              setParsedSections([...liveSections]);
            }
          }
          lineBuf = '';
          return;
        }

        if (liveSections.length === 0) return; // still before first heading

        // Stream character into the last (active) section's content
        const last = liveSections[liveSections.length - 1];
        last.content += ch;

        // Throttle React renders: schedule a flush on next animation frame,
        // coalescing rapid token bursts into single renders per frame.
        if (!parseTimerRef.current) {
          parseTimerRef.current = window.requestAnimationFrame(() => {
            parseTimerRef.current = null;
            setParsedSections([...liveSections]);
          }) as unknown as number;
        }
      };

      // ── Per-token state machine for live think/answer display ──
      // Tags like </think> are often split across multiple tokens (e.g. "</" then "think>").
      // We maintain a persistent `tagBuf` to accumulate partial tag bytes across tokens
      // before we commit them to the appropriate output buffer.
      type TokState = 'thinking' | 'kb_query' | 'answer';
      let tokState: TokState = 'thinking';
      let thinkBuf = '';   // text inside the current active <think> block
      // tagBuf holds a potential partial opening of a tag (e.g. "<", "</", "</th") so we
      // don't emit it prematurely before deciding which tag it belongs to.
      let tagBuf = '';

      // The tags we need to detect (lowercase). Max tag length = 11 chars "</kb_query>".
      const MAX_TAG_LEN = 11;

      const flushTagBuf = () => {
        // tagBuf didn't complete a recognised tag — emit it to the current output buffer.
        if (!tagBuf) return;
        const tb = tagBuf;
        tagBuf = '';
        if (tokState === 'thinking') {
          thinkBuf += tb;
          setThinkingText(thinkBuf);
        } else if (tokState === 'answer') {
          for (const ch of tb) pushAnswerChar(ch);
        }
        // kb_query: discard
      };

      const processToken = (tok: string) => {
        accumulated += tok;

        // Feed each character through the state machine with tag-boundary awareness.
        for (let i = 0; i < tok.length; i++) {
          const ch = tok[i];

          // If we're accumulating a potential tag, add to it.
          if (tagBuf.length > 0 || ch === '<') {
            tagBuf += ch;
            const tb = tagBuf.toLowerCase();

            // Check if tagBuf matches or could still match a known tag.
            const tags: Array<[string, () => void]> = [
              ['</think>', () => {
                tagBuf = '';
                if (tokState === 'thinking') tokState = 'answer';
              }],
              ['</thinking>', () => {
                tagBuf = '';
                if (tokState === 'thinking') tokState = 'answer';
              }],
              ['<think>', () => {
                tagBuf = '';
                // Enter thinking state. Don't reset thinkBuf — append to existing
                // text so display is continuous across KB round-trips.
                if (tokState !== 'thinking') { tokState = 'thinking'; }
              }],
              ['<thinking>', () => {
                tagBuf = '';
                if (tokState !== 'thinking') { tokState = 'thinking'; }
              }],
              ['<kb_query>', () => {
                tagBuf = '';
                if (tokState === 'thinking') { tokState = 'kb_query'; }
              }],
              ['</kb_query>', () => {
                tagBuf = '';
                if (tokState === 'kb_query') { tokState = 'thinking'; }
                // Do NOT reset thinkBuf — keep previous thinking text visible
                // while the KB round-trip completes and new thinking resumes.
              }],
            ];

            let matched = false;
            let couldMatch = false;

            for (const [tag, action] of tags) {
              if (tb === tag) { action(); matched = true; break; }
              if (tag.startsWith(tb)) { couldMatch = true; }
            }

            if (matched) continue;

            if (!couldMatch || tagBuf.length >= MAX_TAG_LEN) {
              // tagBuf can't become any known tag — flush it and keep going
              flushTagBuf();
              // The current char was already added to tagBuf and flushed,
              // so nothing more to do for this char.
            }
            continue;
          }

          // Normal character (not inside a potential tag)
          if (tokState === 'thinking') {
            thinkBuf += ch;
            // Batch the setThinkingText call — done at end of token below
          } else if (tokState === 'answer') {
            pushAnswerChar(ch);
          }
          // kb_query: discard
        }

        // Batch UI updates after processing the full token.
        // isThinking is set true here and is only cleared by the useEffect
        // watching parsedSections.length — this prevents the thinking panel
        // from flashing away before sections are actually visible.
        if (tokState === 'thinking' && thinkBuf) {
          setThinkingText(thinkBuf);
          setIsThinking(true);
        }
      };

      await generateCdsStreaming(
        prompt,
        (event: StreamEvent) => {
          switch (event.type) {
            case 'turn_start':
              setTurnCount(event.turn ?? 0);
              // Start a new thinking turn — reset the local buffer but keep
              // thinkingText showing its last value until new content arrives,
              // so the panel never flashes off between turns / KB round-trips.
              thinkBuf = '';
              tokState = 'thinking';
              break;
            case 'token':
              processToken(event.text ?? '');
              break;
            case 'kb_result':
              setKbHits((prev) => {
                const incoming = (event.hits ?? []).map((h) => ({ ...h }));
                const merged = [...prev, ...incoming];
                merged.sort((a, b) => b.score - a.score);
                return merged;
              });
              break;
            case 'done':
              setTurnCount(event.turns ?? 0);
              break;
            case 'error':
              streamErrorRef.current = true;
              setErrorMsg(event.message ?? 'Stream error');
              setPhase('error');
              showSnackbar({ title: t('error', 'Error'), kind: 'error', subtitle: event.message ?? 'Stream error' });
              break;
          }
        },
        controller.signal,
      );

      if (controller.signal.aborted) return;
      // Cancel any pending rAF flush
      if (parseTimerRef.current) {
        cancelAnimationFrame(parseTimerRef.current as unknown as number);
        parseTimerRef.current = null;
      }

      // If phase was already set to 'error' by the event handler, do not overwrite
      // Use a local ref to track whether we received a stream-level error
      // (The setPhase('error') will have been queued — we need to read from streamErrorRef)
      if (streamErrorRef.current) {
        setIsThinking(false);
        return;
      }

      setIsThinking(false);
      // Do a final full parse from accumulated to ensure any trailing content
      // without a trailing newline is captured, but preserve the live section order.
      const finalSections = parseModelResponse(accumulated);
      // Merge: use live order but fill in final content (handles edge case where
      // last section had no trailing newline and pushAnswerChar missed some chars)
      if (finalSections.length > 0) {
        finalSections.forEach((fs) => {
          const live = liveSections.find(s => s.title.toLowerCase() === fs.title.toLowerCase());
          if (live) live.content = fs.content;
        });
        setParsedSections([...liveSections]);
      }
      setPhase('done');

      showSnackbar({
        title: t('analysisComplete', 'Analysis Complete'),
        kind: 'success',
        subtitle: 'Analysis complete',
      });
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      setErrorMsg(err?.message ?? 'Unknown error');
      setPhase('error');
      showSnackbar({ title: t('error', 'Error'), kind: 'error', subtitle: err?.message });
    }
  }, [patientUuid, t]);

  const visibleHits = showAllHits ? kbHits : kbHits.slice(0, 3);

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

      <div className={styles.body}>
        {/* Patient Card */}
        <Tile className={styles.patientCard}>
          <div className={styles.patientCardHeader}>
            <h5>{t('patientContext', 'Patient Context')}</h5>
            {phase === 'done' && (
              <Button
                kind="primary"
                size="sm"
                renderIcon={WatsonHealthAiResultsHigh}
                onClick={handleAnalyze}
                className={styles.rerunBtn}
              >
                {t('rerunAi', 'Re-run Analysis')}
              </Button>
            )}
          </div>
          {patient && (
            <div className={styles.patientGrid}>
              <div><span className={styles.label}>{t('name', 'Name')}</span><span className={styles.value}>{patient.name?.[0]?.given?.join(' ')} {patient.name?.[0]?.family}</span></div>
              <div><span className={styles.label}>{t('gender', 'Gender')}</span><span className={styles.value}>{patient.gender}</span></div>
              <div><span className={styles.label}>{t('birthDate', 'Birth Date')}</span><span className={styles.value}>{patient.birthDate}</span></div>
            </div>
          )}
        </Tile>

        {/* Action Button — visible only before streaming starts */}
        {(phase === 'idle' || phase === 'building' || phase === 'error') && (
          <Button
            className={styles.actionBtn}
            kind="primary"
            size="lg"
            renderIcon={WatsonHealthAiResultsHigh}
            onClick={handleAnalyze}
            disabled={phase === 'building'}
          >
            {phase === 'building'
              ? t('buildingCase', 'Building case...')
              : t('getAiInsight', 'Get AI Insight')}
          </Button>
        )}

        {/* Building indicator */}
        {phase === 'building' && (
          <Tile className={styles.analyzingCard}>
            <InlineLoading description="" />
            <div>
              <h5>{t('buildingCaseTitle', 'Preparing patient case...')}</h5>
              <p>{t('buildingCaseDetail', 'Collecting encounters and observations from OpenMRS.')}</p>
            </div>
          </Tile>
        )}

        {/* Live thinking panel — stays mounted while streaming to allow collapse transition */}
        {phase === 'streaming' && thinkingText && (
          <ThinkingPanel text={thinkingText} hiding={!isThinking} />
        )}

        {/* Error */}
        {phase === 'error' && errorMsg && (
          <Tile className={styles.errorCard}>
            <WarningFilled size={20} />
            <p>{errorMsg}</p>
          </Tile>
        )}

        {/* Results — shown progressively during streaming AND after done */}
        {(phase === 'streaming' || phase === 'done') && parsedSections.length > 0 && (
          <div className={`${styles.results}${phase === 'streaming' ? ` ${styles.resultsStreaming}` : ''}`}>
            {phase === 'streaming' && (
              <div className={styles.tagRow} style={{ marginBottom: '0.5rem' }}>
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

            {/* KB Evidence Sources — ranked clickable cards */}
            {kbHits.length > 0 && (
              <Tile className={styles.kbFeed}>
                <h6><DataBase size={16} /> {t('evidenceSources', 'Evidence Sources')}</h6>
                {visibleHits.map((hit, i) => (
                  <button
                    key={i}
                    type="button"
                    className={styles.kbEntry}
                    onClick={() => setSelectedHit(hit)}
                    aria-label={t('openEvidence', 'Open evidence: {{title}}', { title: hit.title })}
                  >
                    <div className={styles.kbEntryHeader}>
                      <span className={styles.kbRank}>#{i + 1}</span>
                      <span className={styles.kbTitle}>{hit.title}</span>
                      <Tag type={hit.score > 0.03 ? 'green' : 'gray'} size="sm">
                        {hit.source}
                      </Tag>
                      <span className={styles.kbScore}>{(hit.score * 1000).toFixed(1)}</span>
                    </div>
                  </button>
                ))}
                {kbHits.length > 3 && (
                  <Button
                    kind="ghost"
                    size="sm"
                    className={styles.kbShowMore}
                    onClick={() => setShowAllHits((v) => !v)}
                  >
                    {showAllHits
                      ? t('showLess', 'Show less')
                      : t('showMore', 'Show {{count}} more', { count: kbHits.length - 3 })}
                  </Button>
                )}
              </Tile>
            )}

            {/* Metadata footer */}
            {phase === 'done' && (
              <div className={styles.metaFooter}>
                <Tag type="gray" size="sm">{turnCount} turns</Tag>
                <Tag type="gray" size="sm">{kbHits.length} evidence items</Tag>
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
        )}
      </div>

      {/* Evidence detail modal */}
      {selectedHit && (() => {
        // Extract "title: ..." breadcrumb from the raw content metadata tail
        const titleMatch = selectedHit.content.match(/\btitle:\s*(.+?)(?:\s+uri:|$)/s);
        const docPath = titleMatch ? titleMatch[1].trim() : null;
        return (
          <ComposedModal open onClose={() => setSelectedHit(null)}>
            <ModalHeader title={selectedHit.title} />
            <ModalBody>
              <div className={styles.kbModalMeta}>
                <Tag type="teal" size="sm">{selectedHit.source}</Tag>
                <span className={styles.kbModalScore}>
                  {t('relevanceScore', 'Relevance: {{score}}', { score: (selectedHit.score * 1000).toFixed(1) })}
                </span>
              </div>
              {docPath && (
                <p className={styles.kbModalPath}>{docPath}</p>
              )}
              <div className={styles.kbModalContent}>
                {renderKbContent(selectedHit.content)}
              </div>
            </ModalBody>
            <ModalFooter>
              <Button kind="secondary" onClick={() => setSelectedHit(null)}>
                {t('close', 'Close')}
              </Button>
            </ModalFooter>
          </ComposedModal>
        );
      })()}
    </div>
  );
};

export default CdsWorkspace;
