import React, { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ActionMenuButton, launchWorkspace } from '@openmrs/esm-framework';
import { WatsonHealthAiResultsHigh } from '@carbon/react/icons';

const CdsActionButton: React.FC = () => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => {
    launchWorkspace('clinicdx-cds-workspace');
  }, []);

  return (
    <ActionMenuButton
      getIcon={(props) => <WatsonHealthAiResultsHigh {...props} size={20} />}
      label={t('clinicDxCds', 'ClinicDx CDS')}
      iconDescription={t('clinicDxCdsDesc', 'AI Clinical Decision Support')}
      handler={handleClick}
      type="clinicdx-cds"
    />
  );
};

export default CdsActionButton;
