#!/usr/bin/env python3
import os
import numpy as np
import time
import calendar
from datetime import datetime

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
from prepbufr_obs_builder import PrepbufrObsBuilder
from bufr.encoders import netcdf

MAPPING_PATH = map_path('prepbufr_sonde.yaml')
FILE_ENCODER_DICT = {'netcdf': netcdf.Encoder}

class AdpupaPrepbufrObsBuilder(PrepbufrObsBuilder):
    """
    A builder class to generate ADPUPA observations from ADPUPA prepBUFR input.
    Modified from NCEP SPOC for NASA GMAO observation processing
    Adds GMAO blacklist and correction to drifter timestamps to match GSI read_prepbufr.f90 
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def create_obs_file(self, input, output, type='netcdf', append=False):
        """
        Create an observation file from the input data. Override this method if you want to
        customize the file creation process or if you need a different function signature (ex: you
        need to pass multiple input files). add_main_functions will copy the function signature.

        :param input: Input path to the BUFR file.
        :param output: Output file name
        :param type: Data type to encode into (optional)
        :param append: Add to the file if it exists or create a new file. (optional)
        """

        comm = bufr.mpi.Comm("world")
        self.log.comm = comm

        sonde_container = self.make_obs(comm, input)
        pibal_container = self.make_obs(comm, input)


        # Gather and encode data 
        rank = comm.rank()
        size = comm.size()

        self.log.info("Container with no cagegories defined - encoding the container at rank 0.")
        sonde_container.gather(comm)
        pibal_container.gather(comm)


        if comm.rank() == 0:
            typ=sonde_container.get('observationType')
            sonde_container.apply_mask((typ==120)|(typ==220))
            FILE_ENCODER_DICT[type](self.description).encode(sonde_container, output.replace("temp","sonde"), append)
            pibal_container.apply_mask(typ==221)
            FILE_ENCODER_DICT[type](self.description).encode(pibal_container, output.replace("temp","pibal"), append)

        self.log.info(f'Return the encoded data')


    def make_obs(self, comm, input_path):

        # Get container from mapping file first
        self.log.info(f'Get container from bufr')
        container = super().make_obs(comm, input_path)

        self.log.debug(f'container list (original): {container.list()}')
        for cat in container.all_sub_categories(): 
           self.log.debug(f'Perform DateTime calculation and correction for drifting obs')
           hrdr = container.get('obsTimeMinusCycleTime',cat)
           self._replace_timestamp(container, self._get_reference_time(input_path),catID=cat)
           self._correct_drift_times(container, self._get_reference_time(input_path),catID=cat)

           self.log.debug(f'Make an array of 0s for ObsSubType')
           obsSubType = np.zeros(hrdr.shape, dtype=np.int32)
           self.log.debug(f' obsSubType min/max =  {obsSubType.min()} {obsSubType.max()}')

           self.log.debug(f'Perform stationPressure, stationPressureQM calculations')
           pbdlcat = container.get('prepbufrDataLevelCategory',cat)
           pob = container.get('pressure',cat)
           pqm = container.get('pressureQualityMarker',cat)
           poe = container.get('pressureError',cat)

           station_pressure_blacklist=self._get_blacklist(container,'ps',catID=cat)
           station_pressure = self._compute_conditional_array(pob, ((pbdlcat == 0)  & (~station_pressure_blacklist)))
           station_pressureQM = self._compute_conditional_array(pqm,((pbdlcat == 0)  & (~station_pressure_blacklist)))
           station_pressureError = self._compute_conditional_array(poe, ((pbdlcat == 0)  & (~station_pressure_blacklist)))

           self.log.debug(f'Perform airTemperature, airTemperatureQM, and airTemperatureError calculations')
           tpc = container.get('temperatureEventCode',cat)
           tob = container.get('airTemperature',cat)
           tobqm = container.get('airTemperatureQualityMarker',cat)
           toboe = container.get('airTemperatureError',cat)
           air_temperature_blacklist=self._get_blacklist(container,'t',catID=cat)

           air_temperature = self._compute_conditional_array(tob, (tpc >= 1) & (tpc < 8) &  (~air_temperature_blacklist))
           air_temperatureQM = self._compute_conditional_array(tobqm, (tpc >= 1) & (tpc < 8) &  (~air_temperature_blacklist))
           air_temperatureError = self._compute_conditional_array(toboe, (tpc >= 1) & (tpc < 8) &  (~air_temperature_blacklist))

           self.log.debug(f'Perform virtualTemperature, virtualTemperatureQM, and virtualTemperatureError calculations')
           virtual_temperature_blacklist=self._get_blacklist(container,'tv',catID=cat)

           virtual_temperature = self._compute_conditional_array(tob, (tpc == 8)  & (~virtual_temperature_blacklist))
           virtual_temperatureQM = self._compute_conditional_array(tobqm, (tpc == 8)  & (~virtual_temperature_blacklist))
           virtual_temperatureError = self._compute_conditional_array(toboe, (tpc == 8)  & (~virtual_temperature_blacklist))

           self.log.debug(f'Perform eastwind,eastwindQM, and eastwindError calculations')
           uob = container.get('windEastward',cat)
           vob = container.get('windNorthward',cat)
           wobqm = container.get('windQualityMarker',cat)
           woboe = container.get('windError',cat)

           wind_blacklist=self._get_blacklist(container,'uv',catID=cat)
           wind_eastward = self._compute_conditional_array(uob, (~wind_blacklist))
           wind_northward = self._compute_conditional_array(vob, (~wind_blacklist))
           wind_QC = self._compute_conditional_array(wobqm, (~wind_blacklist))
           wind_Error = self._compute_conditional_array(woboe, (~wind_blacklist))


           self.log.debug(f'Perform specifichumidity,specifichumidityQM,specifichumidityError calculations')
           qob = container.get('specificHumidity',cat)
           qobqm = container.get('specificHumidityQualityMarker',cat)
           qoboe = container.get('specificHumidityError',cat)

           specific_humidity_blacklist=self._get_blacklist(container,'q',catID=cat)
           specific_humidity = self._compute_conditional_array(qob, (~specific_humidity_blacklist))
           specific_humidityQC = self._compute_conditional_array(qobqm, (~specific_humidity_blacklist))
           specific_humidityError = self._compute_conditional_array(qoboe, (~specific_humidity_blacklist))

           self.log.debug(f'Update variables into container')
           container.replace('airTemperature', air_temperature,cat)
           container.replace('airTemperatureQualityMarker', air_temperatureQM,cat)
           container.replace('airTemperatureQualityMarker', air_temperatureQM,cat)

           container.replace('virtualTemperature', virtual_temperature,cat)
           container.replace('virtualTemperatureQualityMarker', virtual_temperatureQM,cat)
           container.replace('virtualTemperatureQualityMarker', virtual_temperatureQM,cat)

           container.replace('specificHumidity', specific_humidity,cat)
           container.replace('specificHumidityQualityMarker', specific_humidityQC,cat)
           container.replace('specificHumidityQualityMarker', specific_humidityQC,cat)

           container.replace('windEastward', wind_eastward,cat)
           container.replace('windNorthward', wind_northward,cat)
           container.replace('windQualityMarker', wind_QC,cat)
           container.replace('windError', wind_Error,cat)


           self.log.debug(f'Add new/derived variables into container')
           ydr_paths = container.get_paths('latitude',cat)
           container.add('stationPressure', station_pressure, ydr_paths,cat)
           container.add('stationPressureQualityMarker', station_pressureQM, ydr_paths,cat)
           container.add('obsSubType', obsSubType, ydr_paths,cat)


           container.apply_mask(~container.get('latitude',cat).mask,cat)

        self.log.debug(f'container list (updated): {container.list()}')
         
        #container.apply_mask(~container.get('latitude').mask)

        return container

    def _make_description(self):
        description = super()._make_description()

        description.add_variables([
            {
                'name': 'ObsValue/stationPressure',
                'source': 'stationPressure',
                'units': 'Pa',
                'longName': 'Station Pressure',
            },
            {
                'name': 'QualityMarker/stationPressure',
                'source': 'stationPressureQualityMarker',
                'units': '',
                'longName': 'Station Pressure Quality Marker',
            },
            {
                'name': 'ObsSubType/stationPressure',
                'source': 'obsSubType',
                'longName': 'Observation SubType',
            },
            {
                'name': 'ObsSubType/airTemperature',
                'source': 'obsSubType',
                'longName': 'Observation SubType',
            },
            {
                'name': 'ObsSubType/virtualTemperature',
                'source': 'obsSubType',
                'longName': 'Observation SubType',
            },
            {
                'name': 'ObsSubType/specificHumidity',
                'source': 'obsSubType',
                'longName': 'Observation SubType',
            },
            {
                'name': 'ObsSubType/windEastward',
                'source': 'obsSubType',
                'longName': 'Observation SubType',
            },
            {
                'name': 'ObsSubType/windNorthward',
                'source': 'obsSubType',
                'longName': 'Observation SubType',
            }
        ])

        return description

# Add main functions create_obs_file or create_obs_group
add_main_functions(AdpupaPrepbufrObsBuilder)
