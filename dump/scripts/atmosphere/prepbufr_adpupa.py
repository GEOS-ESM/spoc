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

MAPPING_PATH = map_path('prepbufr_adpupa.yaml')
FILE_ENCODER_DICT = {'netcdf': netcdf.Encoder}

class AdpupaPrepbufrObsBuilder(PrepbufrObsBuilder):
    """
    A builder class to generate pibal and sonde obs spaces  from ADPUPA prepBUFR subsets
    Modified from NCEP SPOC for NASA GMAO observation processing
    Adds GMAO blacklist and correction to drifter timestamps to match GSI read_prepbufr.f90 
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__),blacklist_path='/discover/nobackup/jemccurr/bufr-query/custom_tests/gmao_global_blacklist.txt')


    def make_obs(self, comm, input_path):

        # Get container from mapping file first
        self.log.info(f'Get container from bufr')
        container = super().make_obs(comm, input_path)
        self.log.debug(f'container list (original): {container.list()}')
        active_subcats=[]
        #loop through categories - in this case KX values from bufr typ field
        for cat in container.all_sub_categories(): 

           self.log.debug(f'Perform DateTime calculation and correction for drifting obs')
           hrdr = container.get('obsTimeMinusCycleTime',cat)
           #skip category if empty
           if hrdr.size>0:
               active_subcats.append(cat[0])
           else:
               continue 
           #make timestamp and drift corrections
           #self._replace_timestamp(container, self._get_reference_time(input_path),catID=cat)
           self._correct_drift_times(container, self._get_reference_time(input_path),catID=cat)

           #correct launchtime to int -  needed for sonde variational yaml
           #dhr = container.get('launchTimeMinusCycleTime',cat).astype(np.int64)


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
           virtual_temperature_blacklist=self._get_blacklist(container,'t',catID=cat)

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

           #container.replace('launchTimeMinusCycleTime',dhr,cat)

           self.log.debug(f'Add new/derived variables into container')
           ydr_paths = container.get_paths('latitude',cat)
           container.add('stationPressure', station_pressure, ydr_paths,cat)
           container.add('stationPressureQualityMarker', station_pressureQM, ydr_paths,cat)
           container.add('obsSubType', obsSubType, ydr_paths,cat)

           new_latitudes=self._filter_identical(container,catID=cat) #identify identical obs and add to mask 
           container.replace('latitude',new_latitudes,cat)
           container.apply_mask(~container.get('latitude',cat).mask,cat) #remove empty subsets + identical obs

        self.log.debug(f'container list (updated): {container.list()}')
        ##############################################
        #refactor obstype split into sonde/pibal split
        #sonde = kx 120,220,132,232 pibal = kx 221
        ##############################################
        category_map =  {'splits/obsType': ['sonde', 'pibal']}
        new_container = bufr.DataContainer(category_map)
        # Add pibal data
        for var_name in container.list():
              new_container.add(var_name,
                      container.get(var_name, ['pibal']),
                      container.get_paths(var_name, ['pibal']),
                      ['pibal'])
        # Add sonde data
        available_sonde=['sonde_mass','recon_mass','sonde_wind','recon_wind']
        active_sonde=[cat for cat in active_subcats if cat in available_sonde]
        for var_name in container.list():
           var = np.concatenate([container.get(var_name, [cat]) for cat in active_sonde], axis=0)
           new_container.add(var_name,
                      var,
                      container.get_paths(var_name, [active_sonde[0]]),
                      ['sonde'])
        return new_container

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
