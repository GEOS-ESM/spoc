#!/usr/bin/env python3
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
from bufr.obs_builder import nprocs_per_task, add_dummy_variable
from bufr.transforms import compute_solar_angles
from datetime import datetime

MAPPING_PATH = map_path('radiance_ssmis.yaml')


class BufrSsmisObsBuilder(ObsBuilder):
    """
    Class for building observations from ssmis BUFR data.

    This class extends `ObsBuilder` to include specific logic for processing
    SSMIS data such as solar angles and satellite ascending/descending orbits.

    :param mapping_path: Path to the mapping file.
    :type mapping_path: str
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):

        # Get container from mapping file first
        self.log.info('Get container from bufr')
        container = super().make_obs(comm, input_path)

        self.log.debug(f'container list (original): {container.list()}')
        self.log.debug(f'all_sub_categories =  {container.all_sub_categories()}')
        self.log.debug(f'category map =  {container.get_category_map()}')

        # Add new/derived data into container
        for cat in container.all_sub_categories():
            nlocs = container.get('latitude', cat).size
            if nlocs == 0:
                self.log.warning(f"Writing empty file for empty category {cat[0]}")
            self._add_sensor_zenith_and_solar_angles(container, cat)
##            self._add_satellite_ascend_descent_orbit(container, cat)
            self._add_gsi_compatible_ascend_descent(container, cat)

        # Check
        self.log.debug(f'container list (updated): {container.list()}')
        self.log.debug(f'all_sub_categories {container.all_sub_categories()}')

        for cat in container.all_sub_categories():
            self.log.warning(f"cat={cat} nlat={container.get('latitude',cat).size} "
                             f"nchn={container.get('sensorChannelNumber',cat).size}")

        self.log.warning(f"category map = {container.get_category_map()}")
        self.log.warning("driver template output_file = radiance_ssmis_{splits/satId}.nc")

        return container

    def _make_description(self):
        description = super()._make_description()

        return description


    def _add_gsi_compatible_ascend_descent(self, container, category):
        """
        Determine satellite orbit type (ascending or descending) based on latitude changes.

        :param container: Observation data container.
        :type container: Container
        :param category: Data category to process.
        :type category: str
        """

        satId = container.get('satelliteId', category)

        if not satId.size:
            add_dummy_variable(container, 'satelliteAscendingFlag', category, 'satelliteId')
            return
        else:
            # Get data from container
            time = container.get('timestamp', category)
            lat = container.get('latitude', category)
            lon = container.get('longitude', category)
            fov = container.get('fieldOfViewNumber', category)

            num_obs = len(time)
            node_inout = np.zeros(num_obs, dtype=np.int32)

            sort_indices = np.argsort(time)

            MAX_FOV = 60
            AS_NODE = 1
            DS_NODE = -1

    # Determine scan line index for each observation
            scanline = np.zeros(num_obs, dtype=int)
            t1 = time[sort_indices[0]]
            nscan = 1
            scanline[sort_indices[0]] = nscan

            for iobs in range(1, num_obs):
                t2 = time[sort_indices[iobs]]
                tdiff = t2 - t1
                if tdiff >= 0.9000:
                    nscan += 1
                t1 = t2
                scanline[sort_indices[iobs]] = nscan

            max_scan = np.max(scanline)

    # Allocate and initialize 2D arrays (shape: [MAX_FOV, max_scan])
            missing_value = bufr.get_missing_value(np.float64)

            latitude = np.full((MAX_FOV, max_scan), missing_value, dtype=np.float64)
            scanline_back = np.full((MAX_FOV, max_scan), -1, dtype=int)
            nodeinfo = np.full((MAX_FOV, max_scan), 1000, dtype=int)

    # Map 1D observations to 2D scanline grid (adjusting 1-based FOV/scanline to 0-based indexing)
            for iobs in range(num_obs):
                ifov_idx = fov[iobs] - 1          # 1-based fov -> 0-based index
                iscan_idx = scanline[iobs] - 1    # 1-based scanline -> 0-based index
                latitude[ifov_idx, iscan_idx] = lat[iobs]
                scanline_back[ifov_idx, iscan_idx] = iobs  # 0-based observation index

    # Determine ascending/descending node for each scanline
            for iscan in range(max_scan - 1):
                for ifov in range(MAX_FOV):
            # Check if both current and next scanlines have valid observations at this FOV
                    if scanline_back[ifov, iscan] >= 0 and scanline_back[ifov, iscan + 1] >= 0:
                        dlat = latitude[ifov, iscan + 1] - latitude[ifov, iscan]
                        if dlat < 0.0:
                            nodeinfo[:, iscan] = DS_NODE
                        else:
                            nodeinfo[:, iscan] = AS_NODE
                        break  # Cycle loop1 (moves to next scan line)

    # Set last scanline's node equal to the second-to-last
            if max_scan > 1:
                nodeinfo[:, max_scan - 1] = nodeinfo[:, max_scan - 2]

    # Write back node status to node_inout array
            for iscan in range(max_scan):
                for ifov in range(MAX_FOV):
                    iobs = scanline_back[ifov, iscan]
                    if iobs >= 0:
                        node_inout[iobs] = nodeinfo[ifov, iscan]

            paths = container.get_paths('fieldOfViewNumber', category)
            self.log.debug(f'paths = {paths}')

        container.add('satelliteAscendingFlag', node_inout, paths, category)

    def _add_satellite_ascend_descent_orbit(self, container, category):
        """
        Determine satellite orbit type (ascending or descending) based on latitude changes.

        :param container: Observation data container.
        :type container: Container
        :param category: Data category to process.
        :type category: str
        """

        satId = container.get('satelliteId', category)

        if not satId.size:
            add_dummy_variable(container, 'satelliteAscendingFlag', category, 'satelliteId')
            return
        else:
            # Get data from container
            # ephemeris data - latitude values in order of time

            first_lat = container.get('latitude1', category)
            self.log.debug(f'first_lat min/max = {first_lat.min()} {first_lat.max()}')
            second_lat = container.get('latitude2', category)
            self.log.debug(f'second_lat min/max = {second_lat.min()} {second_lat.max()}')
            fovn = container.get('fieldOfViewNumber', category)
            self.log.debug(f'fovn min/max = {fovn.min()} {fovn.max()}')

            # Determine ascending/descending mode
            # Compare latitude between the first and second records
            orbit = np.where(second_lat > first_lat, 1, -1).astype(np.int32)

            self.log.debug(f'orbit min/max = {orbit.min()} {orbit.max()}')

            paths = container.get_paths('fieldOfViewNumber', category)
            self.log.debug(f'paths = {paths}')

        container.add('satelliteAscendingFlag', orbit, paths, category)

    def _add_sensor_zenith_and_solar_angles(self, container, category):
        """
        Compute and add solar zenith and azimuth angles to the observation container.

        :param container: Observation data container.
        :type container: Container
        :param category: Data category to process.
        :type category: str
        """

        satId = container.get('satelliteId', category)

        if not satId.size:
            dummy_mappings = [
                ('solarZenithAngle', 'latitude'),
                ('solarAzimuthAngle', 'latitude'),
                ('sensorZenithAngle', 'latitude'),
                ('sensorAzimuthAngle', 'latitude')
            ]
            for target_var, source_var in dummy_mappings:
                add_dummy_variable(container, target_var, category, source_var)

            return

        # Prepare input arrays
        unix_times = container.get('timestamp', category)
        latitudes = container.get('latitude', category)
        longitudes = container.get('longitude', category)
        self.log.debug(f'latitudes min/max = {latitudes.min()} {latitudes.max()}')
        self.log.debug(f'longitudes min/max = {longitudes.min()} {longitudes.max()}')
        self.log.debug(f'unix_times min/max = {unix_times.min()} {unix_times.max()}')

        # Calculate solar angles
        zenith_angles, azimuth_angles = compute_solar_angles(latitudes, longitudes, unix_times)

        self.log.debug(f'zenith_angles min/max = {zenith_angles.min()} {zenith_angles.max()}')
        self.log.debug(f'azimuth_angles min/max = {azimuth_angles.min()} {azimuth_angles.max()}')

        # Add solar angles
        paths = container.get_paths('latitude', category)
        self.log.debug(f'paths = {paths}')
        container.add('solarZenithAngle', zenith_angles, paths, category)
        container.add('solarAzimuthAngle', azimuth_angles, paths, category)

        # Add sensor angles
        sensor_zenith = np.full_like(latitudes, 53.0)
        sensor_azimuth = np.zeros_like(latitudes, dtype=np.float32)
        container.add('sensorZenithAngle', sensor_zenith, paths, category)
        container.add('sensorAzimuthAngle', sensor_azimuth, paths, category)

# Add main functions create_obs_file or create_obs_group
add_main_functions(BufrSsmisObsBuilder)
