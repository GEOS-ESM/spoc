#!/usr/bin/env python3
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('radiance_avhrr.yaml')


class BufrGsrasrObsBuilder(ObsBuilder):

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        # Get container from mapping file
        self.log.info('Get container from bufr')
        container = super().make_obs(comm, input_path)
        self.log.debug(f'Container list (original): {container.list()}')
        self.log.debug(f'All_sub_categories =  {container.all_sub_categories()}')
        self.log.debug(f'Category map = {container.get_category_map()}')

        cut_spot = np.float32(11.0)
        ngac = np.float32(409.0)
        nfov = np.float32(90.0)
        scan_start=np.float32(-52.612)
        scan_inc=np.float32(1.182)
        dfov = (ngac - 2 * cut_spot - 1) / nfov
        self.log.info(f'dfov= {dfov}')

        for cat in container.all_sub_categories():
            self.log.debug(f'category: {cat}')

            sensor_channel_num = container.get('sensorChannelNumber', cat)

            sensor_channel_num[:, 0] = 3
            sensor_channel_num[:, 1] = 4
            sensor_channel_num[:, 2] = 5
            container.replace('sensorChannelNumber',sensor_channel_num,cat)

            # Get the fieldOfViewNumber array from the container (for reference)
            fieldOfViewNumber = container.get('fieldOfViewNumber', cat)
            paths = container.get_paths('fieldOfViewNumber', cat)

            # make the sensorAzimuthAngle and solarAzimuthAngle array with all missing values
            missing_value = bufr.get_missing_value(np.float32)

            sensorAzimuthAngle = np.full_like(fieldOfViewNumber, fill_value=missing_value, dtype=np.float32)
            solarAzimuthAngle = np.full_like(fieldOfViewNumber, fill_value=missing_value, dtype=np.float32)
            scan_pos = np.zeros_like(fieldOfViewNumber, dtype=np.float32)
            scan_angle = np.zeros_like(fieldOfViewNumber, dtype=np.float32)

            def fortran_nint(x):
                return np.floor(x + 0.5)

            diff = fieldOfViewNumber - cut_spot
            ratio = diff / dfov

            # Use fmod for exact Fortran 'mod' behavior with negative numbers
            mask = np.fmod(diff, dfov) < (0.5 * dfov)
            scan_pos = np.where(mask,
                    fortran_nint(ratio) + np.float32(1.0),
                    fortran_nint(ratio)).astype(np.float32)

            # np.clip restricts values to a range: [min, max]
            # Here we keep the minimum as is, but cap the maximum at nfov
            scan_pos = np.clip(scan_pos, None, nfov)

            ifov = fortran_nint(scan_pos)

            # Calculate the scan angle for all indices at once
            # This replaces the entire loop with a single array operation
            scan_angle = scan_start + (ifov - 1) * scan_inc

            # add the sensorAzimuthAngle array to the bufr container
            container.add('sensorAzimuthAngle', sensorAzimuthAngle, paths, cat)
            # add the solarAzimuthAngle array to the bufr container
            container.add('solarAzimuthAngle', solarAzimuthAngle, paths, cat)
            # add the sensorScanPosition array to the bufr container
            container.add('sensorScanPosition', scan_pos, paths, cat)
            # add the sensorViewAngle array to the bufr container
            container.add('sensorViewAngle', scan_angle, paths, cat)

            satId = container.get('satelliteId', cat)

            if not np.any(satId):
                self.log.warning(f'Category {cat[0]} does not exist in input file')
                continue  # Skip invalid category

        # Final container state
        self.log.debug(f'Container list (updated): {container.list()}')

        return container

    def _make_description(self):
        description = super()._make_description()

        # Add the variables to the description for export
        description.add_variables([
            {
                'name': 'MetaData/sensorAzimuthAngle',
                'source': 'sensorAzimuthAngle',
                'units': 'degrees',
                'longName': 'Sensor Azimuth Angle',
            },
            {
                'name': 'MetaData/solarAzimuthAngle',
                'source': 'solarAzimuthAngle',
                'units': 'degrees',
                'longName': 'Solar Azimuth Angle',
            },
            {
                'name': 'MetaData/sensorScanPosition',
                'source': 'sensorScanPosition',
                'longName': 'Sensor Scan Position',
            },
            {
                'name': 'MetaData/sensorViewAngle',
                'source': 'sensorViewAngle',
                'units': 'degrees',
                'longName': 'Sensor Scan Angle',
            },
        ])
        return description


add_main_functions(BufrGsrasrObsBuilder)
