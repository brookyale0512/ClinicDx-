import React, { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { ActionMenuButton, launchWorkspace } from '@openmrs/esm-framework';
import { ImageSearch } from '@carbon/react/icons';

const ImagingActionButton: React.FC = () => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => {
    launchWorkspace('clinicdx-imaging-workspace');
  }, []);

  return (
    <ActionMenuButton
      getIcon={(props) => <ImageSearch {...props} size={20} />}
      label={t('clinicDxImaging', 'ClinicDx Imaging')}
      iconDescription={t('clinicDxImagingDesc', 'DICOM Image Analysis')}
      handler={handleClick}
      type="clinicdx-imaging"
    />
  );
};

export default ImagingActionButton;
