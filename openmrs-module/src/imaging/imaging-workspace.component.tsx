import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Button,
  Tile,
  Layer,
  Tag,
  InlineLoading,
} from '@carbon/react';
import { ImageSearch, Upload, TrashCan, DocumentBlank } from '@carbon/react/icons';
import { usePatient, useFeatureFlag, showSnackbar } from '@openmrs/esm-framework';
import styles from './imaging-workspace.scss';

interface ImagingWorkspaceProps {
  patientUuid: string;
  closeWorkspace: (options?: { ignoreChanges?: boolean }) => void;
  promptBeforeClosing: (hasUnsavedChanges: () => boolean) => void;
}

interface DicomFile {
  id: string;
  file: File;
  fileName: string;
  fileSize: string;
  timestamp: Date;
}

/** Maximum DICOM file size in MB (S-4) */
const MAX_FILE_SIZE_MB = 500;
/** Maximum number of files at once (S-4) */
const MAX_FILE_COUNT = 20;

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

const ImagingWorkspace: React.FC<ImagingWorkspaceProps> = ({
  patientUuid,
  closeWorkspace,
  promptBeforeClosing,
}) => {
  const { t } = useTranslation();
  const { patient, isLoading: isLoadingPatient } = usePatient(patientUuid);
  const isImagingEnabled = useFeatureFlag('clinicdx-imaging');

  const [dicomFiles, setDicomFiles] = useState<DicomFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<DicomFile | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    promptBeforeClosing(() => dicomFiles.length > 0);
  }, [promptBeforeClosing, dicomFiles.length]);

  const addFiles = useCallback((files: FileList | File[]) => {
    const fileArray = Array.from(files);

    // S-4: file count limit
    if (dicomFiles.length + fileArray.length > MAX_FILE_COUNT) {
      showSnackbar({
        title: t('error', 'Error'),
        kind: 'error',
        subtitle: t('tooManyFiles', 'Maximum {{max}} files allowed', { max: MAX_FILE_COUNT }),
      });
      return;
    }

    const validFiles: DicomFile[] = [];
    for (const file of fileArray) {
      // S-4: file size limit
      if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        showSnackbar({
          title: t('error', 'Error'),
          kind: 'error',
          subtitle: t('fileTooLarge', 'File {{name}} exceeds {{max}}MB limit', { name: file.name, max: MAX_FILE_SIZE_MB }),
        });
        continue;
      }
      validFiles.push({
        id: `dcm-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        file,
        fileName: file.name,
        fileSize: formatFileSize(file.size),
        timestamp: new Date(),
      });
    }

    if (validFiles.length === 0) return;

    setDicomFiles((prev) => [...validFiles, ...prev]);
    if (validFiles.length > 0) setSelectedFile(validFiles[0]);

    showSnackbar({
      title: t('filesAdded', 'Files Added'),
      kind: 'success',
      subtitle: `${validFiles.length} DICOM file${validFiles.length > 1 ? 's' : ''} ready for analysis`,
    });
  }, [t, dicomFiles.length]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [addFiles]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const deleteFile = useCallback((id: string) => {
    setDicomFiles((prev) => prev.filter((f) => f.id !== id));
    if (selectedFile?.id === id) setSelectedFile(null);
  }, [selectedFile]);

  if (isLoadingPatient) {
    return (
      <div className={styles.loadingContainer}>
        <InlineLoading description={t('loadingPatient', 'Loading patient data...')} />
      </div>
    );
  }

  // Q-3: Feature flag guard
  if (!isImagingEnabled) {
    return (
      <div className={styles.workspaceContainer}>
        <Tile className={styles.underDevTile}>
          <h4>{t('clinicDxImaging', 'ClinicDx Imaging')}</h4>
          <p>{t('comingSoon', 'This workspace is under development.')}</p>
        </Tile>
      </div>
    );
  }

  return (
    <div className={styles.workspaceContainer}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <ImageSearch size={24} />
          <h4 className={styles.title}>{t('clinicDxImaging', 'ClinicDx Imaging')}</h4>
          <Tag type="blue" size="sm">{t('aiPowered', 'AI-Powered')}</Tag>
        </div>
        <p className={styles.subtitle}>
          {t('imagingSubtitle', 'Upload DICOM files for AI-powered medical image analysis')}
        </p>
      </div>

      <Layer className={styles.content}>
        <Tile className={styles.patientInfo}>
          <h5>{t('patientContext', 'Patient Context')}</h5>
          {patient && (
            <div className={styles.patientDetails}>
              <p><strong>{t('name', 'Name')}:</strong> {patient.name?.[0]?.given?.join(' ')} {patient.name?.[0]?.family}</p>
              <p><strong>{t('gender', 'Gender')}:</strong> {patient.gender}</p>
              <p><strong>{t('id', 'ID')}:</strong> {patient.id}</p>
            </div>
          )}
        </Tile>

        {/* A-4: drop zone keyboard accessibility */}
        <div
          className={`${styles.dropZone} ${isDragOver ? styles.dragOver : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label={t('uploadDicom', 'Upload DICOM Files')}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
        >
          <Upload size={48} className={styles.dropIcon} />
          <p className={styles.dropTitle}>{t('uploadDicom', 'Upload DICOM Files')}</p>
          <p className={styles.dropHint}>
            {t('dropHint', 'Drag and drop DICOM files here, or click to browse')}
          </p>
          <p className={styles.dropFormats}>
            {t('supportedFormats', '.dcm, .dicom, .DCM — or any DICOM-compatible file')}
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".dcm,.dicom,.DCM,application/dicom"
            multiple
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>

        {dicomFiles.length > 0 && (
          <div className={styles.fileList}>
            <h5 className={styles.listTitle}>
              {t('dicomFiles', 'DICOM Files')} ({dicomFiles.length})
            </h5>
            {dicomFiles.map((dcm) => (
              <Tile
                key={dcm.id}
                className={`${styles.fileItem} ${selectedFile?.id === dcm.id ? styles.selected : ''}`}
                onClick={() => setSelectedFile(dcm)}
              >
                <div className={styles.fileRow}>
                  <DocumentBlank size={32} className={styles.fileIcon} />
                  <div className={styles.fileInfo}>
                    <span className={styles.fileName}>{dcm.fileName}</span>
                    <span className={styles.fileMeta}>
                      {dcm.fileSize} — {dcm.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  <Button
                    kind="danger--ghost"
                    size="sm"
                    hasIconOnly
                    renderIcon={TrashCan}
                    iconDescription={t('delete', 'Delete')}
                    onClick={(e: React.MouseEvent) => { e.stopPropagation(); deleteFile(dcm.id); }}
                  />
                </div>
              </Tile>
            ))}
          </div>
        )}
      </Layer>
    </div>
  );
};

export default ImagingWorkspace;
