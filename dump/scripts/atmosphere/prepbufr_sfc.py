#!/usr/bin/env python3
import calendar
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import add_main_functions
from prepbufr_obs_builder import PrepbufrObsBuilder, map_path


MAPPING_PATH = map_path('prepbufr_sfc.yaml')


class SurfacePrepbufrObsBuilder(PrepbufrObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__),blacklist_path=os.getenv('blacklist_path'))


    def make_obs(self, comm, input_path):
        """
        Create the ioda sfcshp and adpsfc prepbufr observations:
        - reads values
        - adds sequenceNum
        - adds ObsSubType

        Parameters
        ----------
        comm: object
                The communicator object (e.g., MPI)
        input_path: str
                The input bufr file
        """


        # Get container from mapping file first
        self.log.info('Get container from bufr')
        container = super().make_obs(comm, input_path)
        self.log.debug(f'container list (original): {container.list()}')
        active_subcats=[]
        #loop through categories - in this case observation type from bufr typ field
        for cat in container.all_sub_categories():
           self.log.debug(f'Do DateTime calculation')
           dhr = container.get('obsTimeMinusCycleTime',cat)
           dhr_paths = container.get_paths('obsTimeMinusCycleTime',cat)
           dhr2 = np.array(dhr)
           #no need to replace timestamp if modified in yaml
           #self._replace_timestamp(container, self._get_reference_time(input_path),catID=cat)

           #skip category if empty
           if dhr.size>0:
               active_subcats.append(cat[0])
           else:
               continue

           sid = container.get('stationIdentification',cat)
           sid_paths=container.get_paths('stationIdentification',cat) 
           typ = container.get('observationType',cat)
           typ_paths = container.get_paths('observationType',cat)
           t29 = container.get('observationSubTypeNum',cat)
           t29_paths = container.get_paths('observationSubTypeNum',cat)
           obsSubType = self._compute_obssubtype(typ, t29)
           self.log.debug(f' obsSubType min/max =  {obsSubType.min()} {obsSubType.max()}')

           self.log.debug(f'Do drifting buoy modification')
           #container is updated here instead of end to ensure surface height/additional calculations
           #included in the prepbufr_obsbuilder module get the corrected type info
           typ_drifter_correct=self._compute_drifting_buoys(typ,t29,sid)
           container.replace('observationType',typ_drifter_correct,cat) #container should be updated here
           self.log.debug(f'Do surface ob height correction')
           self._correct_surface_height(container,cat)

           self.log.debug(f'Perform stationPressure, stationPressureQM calculations')
           pbdlcat = container.get('prepbufrDataLevelCategory',cat)
           pob = container.get('stationPressureObsValue',cat)
           pqm = container.get('stationPressureQualityMarker',cat)
           poe = container.get('stationPressureObsError',cat)
           pmsl = container.get('pressureReducedToMeanSeaLevelObsValue',cat) 
           pmq = container.get('pressureReducedToMeanSeaLevelQualityMarker',cat) 
           pmin = container.get('pmoIndicator',cat)
           obs_elv = container.get('height',cat)
           zqm = container.get('heightQualityMarker' ,cat)

           #pressure sanity check
           pob[pob<np.finfo(np.float32).tiny]=pob.fill_value


           #blacklist implementation and simple QC for station pressure + humidity
           station_pressure_blacklist=self._get_blacklist(container,'ps',catID=cat)
           station_pressure = self._compute_conditional_array(pob,((pob>50000) & (pbdlcat == 0) & (~station_pressure_blacklist))).astype(np.float32)
           station_pressureQM = self._compute_conditional_array(pqm,((pob>50000) & (pbdlcat == 0) & (~station_pressure_blacklist)))
           station_pressureError = self._compute_conditional_array(poe,((pob>50000) & (pbdlcat == 0) & (~station_pressure_blacklist)))

           #Apply ship pressure correction
           station_pressure  = self._correct_ship_pressure(typ_drifter_correct,t29,obs_elv,station_pressure,pob.fill_value,pmsl,pmq,pmin)


           self.log.debug(f'Do tsen and tv calculations')
           tpc = container.get('temperatureEventCode',cat)
           tob = container.get('airTemperatureObsValue',cat)
           tob_paths = container.get_paths('airTemperatureObsValue',cat)
           tsen = np.full(tob.shape[0], tob.fill_value)
           tsen = np.where(((tpc >= 1) & (tpc < 8)), tob, tsen)
           tvo = np.full(tob.shape[0], tob.fill_value)
           tvo = np.where((tpc == 8), tob, tvo)

           tobqm = container.get('airTemperatureQualityMarker',cat)
           tsenqm = np.full(tobqm.shape[0], tobqm.fill_value)
           tsenqm = np.where(((tpc >= 1) & (tpc < 8)), tobqm, tsenqm)
           tvoqm = np.full(tobqm.shape[0], tobqm.fill_value)
           tvoqm = np.where((tpc == 8), tobqm, tvoqm)

           toboe = container.get('airTemperatureObsError',cat)
           tsenoe = np.full(toboe.shape[0], toboe.fill_value)
           tsenoe = np.where(((tpc >= 1) & (tpc < 8)), toboe, tsenoe)
           tvooe = np.full(toboe.shape[0], toboe.fill_value)
           tvooe = np.where((tpc == 8), toboe, tvooe)

           air_temperature_blacklist=self._get_blacklist(container,'t',catID=cat)
           virtual_temperature_blacklist=self._get_blacklist(container,'t',catID=cat)

           tsen[air_temperature_blacklist]=tob.fill_value
           tsenqm[air_temperature_blacklist]=tobqm.fill_value
           tsenoe[air_temperature_blacklist]=toboe.fill_value

           tvo[air_temperature_blacklist]=tob.fill_value
           tvoqm[air_temperature_blacklist]=tobqm.fill_value
           tvooe[air_temperature_blacklist]=toboe.fill_value

           self.log.debug(f'Perform northwind,northwindQM calculations')
           uob = container.get('windEastwardObsValue',cat)
           vob = container.get('windNorthwardObsValue',cat)
           wobqm = container.get('windQualityMarker',cat)
           woboe = container.get('windObsError',cat)

           wind_blacklist=self._get_blacklist(container,'uv',catID=cat)
           wind_eastward = self._compute_conditional_array(uob, (~wind_blacklist))
           wind_northward = self._compute_conditional_array(vob, (~wind_blacklist))
           windQC = self._compute_conditional_array(wobqm, (~wind_blacklist))
           windError = self._compute_conditional_array(woboe, (~wind_blacklist))

           self.log.debug(f'Perform specifichumidity,specifichumidityQM calculations')
           qob = container.get('specificHumidityObsValue',cat)
           qobqm = container.get('specificHumidityQualityMarker',cat)
           qoboe = container.get('relativeHumidityObsError',cat)

           specific_humidity_blacklist=self._get_blacklist(container,'q',catID=cat)
           specific_humidity = self._compute_conditional_array(qob,((qob<1000000000)&(~specific_humidity_blacklist)))
           specific_humidityQC = self._compute_conditional_array(qobqm,((qob<1000000000)&(~specific_humidity_blacklist)))
           specific_humidityError = self._compute_conditional_array(qoboe,((qob<1000000000)&(~specific_humidity_blacklist)))

           self.log.debug(f'Update variables in container')
        
           container.replace('stationPressureObsValue',station_pressure,cat)
           container.replace('stationPressureQualityMarker',station_pressureQM,cat)
           container.replace('stationPressureObsError',station_pressureError,cat)

           container.replace('airTemperatureObsValue', tsen,cat)
           container.replace('airTemperatureQualityMarker', tsenqm,cat)
           container.replace('airTemperatureObsError', tsenoe,cat)

           container.replace('virtualTemperatureObsValue', tvo,cat)
           container.replace('virtualTemperatureQualityMarker', tvoqm,cat)
           container.replace('virtualTemperatureObsError', tvooe,cat)

           container.replace('specificHumidityObsValue', specific_humidity,cat)
           container.replace('specificHumidityQualityMarker', specific_humidityQC,cat)
           container.replace('relativeHumidityObsError', specific_humidityError,cat)

           container.replace('windEastwardObsValue', wind_eastward,cat)
           container.replace('windNorthwardObsValue', wind_northward,cat)
           container.replace('windQualityMarker', windQC,cat)
           container.replace('windObsError', windError,cat)

           self.log.debug(f'Add variables to container')
           container.add('sequenceNumber', obsSubType, dhr_paths,cat)
           container.add('obsSubType', obsSubType, dhr_paths,cat)

           #fill preUsage variables
           self._add_usage(container,['stationPressure','airTemperature','virtualTemperature','specificHumidity','windEastward','windNorthward'],catID=cat)
           self._filter_identical(container,catID=cat) #remove identical observations and empty records from container 

        self.log.debug(f'container list (updated): {container.list()}')
        category_map = {'splits/obsType': ['sfcship','sfc']}
        new_container = bufr.DataContainer(category_map)
        ################################################
        #refactor obstype split into surface and ship obs spaces
        ################################################
        # collect ship data
        available_sfcship=['surface_marine_mass_rp','surface_marine_mass_atlas','surface_marine_mass_np','surface_marine_wind_rp','surface_marine_wind_atlas','surface_marine_wind_np']
        active_sfcship=[cat for cat in active_subcats if cat in available_sfcship]
        for var_name in container.list():
           var = np.concatenate([container.get(var_name, [cat]) for cat in active_sfcship], axis=0)
           new_container.add(var_name,
                      var,
                      container.get_paths(var_name, [active_sfcship[0]]),
                      ['sfcship'])
        # collect land surface data
        available_sfc=['surface_land_mass_rp','surface_metar_mass_np','surface_land_wind_rp','surface_metar_wind_np']
        active_sfc=[cat for cat in active_subcats if cat in available_sfc]
        for var_name in container.list():
           var = np.concatenate([container.get(var_name, [cat]) for cat in active_sfc], axis=0)
           new_container.add(var_name,
                      var,
                      container.get_paths(var_name, [active_sfc[0]]),
                      ['sfc'])
        return new_container

    def _correct_ship_pressure(self,typ,t29,oelv,pob_val,pob_fill,pmsl,pmq,pmin):
        #performs correction of reported station pressure for ship (kx180) obs 
        #by swapping reported pressure with pressure reduced to mean sea level 
        #where available and where subtype is between 522 and 525
        mask_pmsl = ((oelv==0)&(typ==180)&(t29 >=522)&(t29 <= 525)&(pmq<4)&(np.rint(pmin)==0))
        new_pressure=np.full(pob_val.shape[0],pob_fill)
        new_pressure=np.where(mask_pmsl,pmsl,pob_val)

        return new_pressure
    def _compute_drifting_buoys(self,typ,t29,sid):
        #changes kx values for drifting buoys idenfitied by subtype and wmo number 
        def check_condition(x):
          try:
            val = int(x)
            return (val % 1000) > 500
          except ValueError:
            return False
        vfunc = np.frompyfunc(check_condition, 1, 1)
        mask_typ = np.isin(typ, [180, 280])
        mask_t29 = np.isin(t29, [562, 564])
        mask_sid = vfunc(sid).astype(bool)
        mask_all = mask_typ & mask_t29 & mask_sid
        new_typ=typ.copy()
        new_typ[mask_all]+=19
        return new_typ

    def _compute_obssubtype(self, typ, t29):
        """ 
        Compute obsSubType group
        
        Parameters:
            typ: observation Type (obsType)
            t29: data dump report type
        
        Returns:
            Masked array of obsSubType values
        """

        mask_typ = np.isin(typ, [180, 280])
        mask_t29 = (t29 > 555) & (t29 < 565)
        obsSubType = np.where(mask_typ & ~mask_t29, 1, 0).astype(np.int32)

        return obsSubType


    def _make_description(self):
        description = super()._make_description()

        description.add_variables([
            {
                'name': 'MetaData/sequenceNumber',
                'source': 'sequenceNumber',
                'longName': 'Sequence Number (Obs Subtype)',
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
            },
            {
                'name': 'PreUseFlag/stationPressure',
                'source': 'stationPressureObsUsage',
                'longName': 'Observation pre-Usage',
            },
            {
                'name': 'PreUseFlag/airTemperature',
                'source': 'airTemperatureObsUsage',
                'longName': 'Observation pre-Usage',
            },
            {
                'name': 'PreUseFlag/virtualTemperature',
                'source': 'virtualTemperatureObsUsage',
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
add_main_functions(SurfacePrepbufrObsBuilder)
