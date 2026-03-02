import React, { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ActionMenuButton, launchWorkspace } from '@openmrs/esm-framework';
import { Camera } from '@carbon/react/icons';

const OcrActionButton: React.FC = () => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => {
    launchWorkspace('clinicdx-ocr-workspace');
  }, []);

  return (
    <ActionMenuButton
      getIcon={(props) => <Camera {...props} size={20} />}
      label={t('clinicDxOcr', 'ClinicDx OCR')}
      iconDescription={t('clinicDxOcrDesc', 'Lab Result Digitization')}
      handler={handleClick}
      type="clinicdx-ocr"
    />
  );
};

export default OcrActionButton;
