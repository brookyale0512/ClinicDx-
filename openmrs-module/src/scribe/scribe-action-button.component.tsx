import React, { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ActionMenuButton, launchWorkspace } from '@openmrs/esm-framework';
import { Microphone } from '@carbon/react/icons';

const ScribeActionButton: React.FC = () => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => {
    launchWorkspace('clinicdx-scribe-workspace');
  }, []);

  return (
    <ActionMenuButton
      getIcon={(props) => <Microphone {...props} size={20} />}
      label={t('clinicDxScribe', 'ClinicDx Scribe')}
      iconDescription={t('clinicDxScribeDesc', 'AI Medical Scribe')}
      handler={handleClick}
      type="clinicdx-scribe"
    />
  );
};

export default ScribeActionButton;
