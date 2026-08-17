#!/usr/bin/env python3
import os
import sys
import argparse
import bufr
import copy
import numpy as np
import numpy.ma as ma
import math
import calendar
import time
from datetime import datetime

import bufr
from bufr.obs_builder import add_main_functions
from prepbufr_obs_builder import PrepbufrObsBuilder, map_path


MAPPING_PATH = map_path('prepbufr_aircraft_wind.yaml')


class AcftProfilesPrepbufrObsBuilder(PrepbufrObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))


    def make_obs(self, comm, input_path):
        """
        Create the ioda acft_profiles prepbufr observations:
        - reads values
        - adds sequenceNum

        Parameters
        ----------
        comm: object
                The communicator object (e.g., MPI)
        input_path: str
                The input bufr file
        mapping_path: str
                The input bufr2ioda mapping file
        cycle_time: str
                The cycle in YYYYMMDDHH format
        """

        # Get container from mapping file first
        self.log.info(f'Get container from bufr')
        container = super().make_obs(comm, input_path)

        self.log.debug(f'container list (original): {container.list()}')
        lon = container.get('longitude')
        lon_paths = container.get_paths('longitude')

        self.log.debug(f'Do DateTime calculation')
        dhr = container.get('obsTimeMinusCycleTime')
        dhr_paths = container.get_paths('obsTimeMinusCycleTime')
        dhr2 = np.array(dhr)
        #timestamp replacement not needed if done in yaml
        #self._replace_timestamp(container, self._get_reference_time(input_path))

        self.log.debug(f'Make an array of 0s for MetaData/sequenceNumber')
        sequenceNum = self._compute_sequence_number(lon)
        self.log.debug(f'sequenceNum min/max =  {sequenceNum.min()} {sequenceNum.max()}')

        self.log.debug(f'Compute Obstypes')
        t_ot = container.get('airTemperatureObservationType')
        q_ot = container.get('specificHumidityObservationType')
        uv_ot = container.get('windObservationType')
        ot_paths = container.get_paths('airTemperatureObservationType')

        airTemperature = container.get('airTemperatureObsValue')
        specificHumidity = container.get('specificHumidityObsValue')
        wind = container.get('windNorthwardObsValue')

        ot_airTemperature = self._compute_typ_other(t_ot)
        ot_specificHumidity = self._compute_typ_other(q_ot)
        ot_wind = self._compute_typ_uv(uv_ot)

        self.log.debug(f'Change IALR to 0.0 if masked for bias correction.')
        ialr = container.get('instantaneousAltitudeRate')
        ialr_paths = container.get_paths('instantaneousAltitudeRate')
        ialr2 = ma.array(ialr)

        ialr_bc = self._compute_ialr_if_masked(uv_ot, ialr2)


        self.log.debug(f'Update variables in container')
        container.replace('instantaneousAltitudeRate', ialr_bc)
        container.replace('airTemperatureObservationType', ot_airTemperature)
        container.replace('specificHumidityObservationType', ot_specificHumidity)
        container.replace('windObservationType', ot_wind)

        self.log.debug(f'Add variables to container')
        container.add('sequenceNumber', sequenceNum, lon_paths)

        # Check
        self.log.debug(f'container list (updated): {container.list()}')

        #fill preUsage variables
        self._add_usage(container,['airTemperature','specificHumidity','windEastward','windNorthward'])
        self._filter_identical(container) #remove identical observations and empty records from container 

        return container

    def _compute_typ_other(self, typ):
        """
        Compute datatype if the variable is not wind.
        Parameters:
            typ: datatype
            var: obsValue variable
        Returns:
            Masked array of the new datatype
        """

        typ_var = copy.deepcopy(typ)
        typ_var[(typ_var > 300) & (typ_var < 400)] -= 200
        typ_var[(typ_var > 400) & (typ_var < 500)] -= 300
        typ_var[(typ_var > 500) & (typ_var < 600)] -= 400

        return typ_var

    def _compute_typ_uv(self, typ):
        """
        Compute datatype if the variable is wind.
        Parameters:
            typ: datatype
            var: obsValue variable
        Returns:
            Masked array of the new datatype
        """

        typ_var = copy.deepcopy(typ)
        typ_var[(typ_var > 300) & (typ_var < 400)] -= 100
        typ_var[(typ_var > 400) & (typ_var < 500)] -= 200
        typ_var[(typ_var > 500) & (typ_var < 600)] -= 300

        return typ_var

    def _compute_ialr_if_masked(self, typ, ialr):
        """
        Compute instantaneousAltitudeRate (IALR) if it is masked.
        Parameters:
            typ: datatype
            ialr: instantaneousAltitudeRate
        Returns:
            Masked array of the updated instantaneousAltitudeRate
        """

        ialr_bc = copy.deepcopy(ialr)
        ialr_bc[(ialr_bc.mask) & (typ >= 330) & (typ < 340)] = 0.0
        ialr_bc.mask[(typ >= 330) & (typ < 340)] = False


        return ialr_bc

    def _compute_sequence_number(self, lon):
        """
        Compute sequenceNumber

        Parameters:
            lon: longitude

        Returns:
            Masked array of sequenceNumber values.
            In this case, array is all 0's.
        """

        sequenceNumber = np.zeros(lon.shape, dtype=np.int32)

        return sequenceNumber


    def _make_description(self):
        description = super()._make_description()

        description.add_variables([
            {
                'name': 'MetaData/sequenceNumber',
                'source': 'sequenceNumber',
                'longName': 'Sequence Number (Obs Subtype)',
            },
            {
                'name': 'PreUseFlag/airTemperature',
                'source': 'airTemperatureObsUsage',
                'longName': 'Observation pre-Usage',
            },
            {
                'name': 'PreUseFlag/specificHumidity',
                'source': 'specificHumidityObsUsage',
                'longName': 'Observation pre-Usage',
            },
            {
                'name': 'PreUseFlag/windEastward',
                'source': 'windEastwardObsUsage',
                'longName': 'Observation pre-Usage',
            },
            {
                'name': 'PreUseFlag/windNorthward',
                'source': 'windNorthwardObsUsage',
                'longName': 'Observation pre-Usage',
            }
        ])

        return description

# Add main functions create_obs_file or create_obs_group
add_main_functions(AcftProfilesPrepbufrObsBuilder)
