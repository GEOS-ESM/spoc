#!/usr/bin/env python3
import os
import re
import numpy as np
import numpy.ma as ma
from pathlib import Path
from datetime import datetime
import bufr
from bufr.obs_builder import ObsBuilder
import json
###############################
def map_path(map_file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, map_file_name)

class PrepbufrObsBuilder(ObsBuilder):
    def __init__(self, mapping_path, log_name=os.path.basename(__file__),blacklist_path=None):
        super().__init__(mapping_path, log_name=log_name)
        self.blacklist_data = {}
        self.driftdat_types=[120,220,221] #kx types with drift information on position and time
        if blacklist_path and os.path.exists(blacklist_path):
            self._load_text_blacklist(blacklist_path)

    def _compute_conditional_array(self, source_array, condition_mask):
        """
        Compute an array where values from source_array are retained
        if condition_mask is True, else fill_value is used.
        """

        result = np.full(source_array.shape, source_array.fill_value)
        result[condition_mask] = source_array[condition_mask]
        return result

    def _get_reference_time(self, input_path) -> np.datetime64:
      """
      Extract ref time using only filename regex
      Replaces the default _get_reference_time method
      from NOAA SPOC, which uses dump file folder names
      """
      filename = Path(input_path).name
      
      # Regex Breakdown:
      # \.         : Matches the literal dot after 'gdas1'
      # (?P<...>)  : Named capture groups for readability
      # \d{2}      : Matches exactly two digits (YY, MM, DD, and HH)
      # \.t        : Matches the literal '.t' prefix for the hour
      # z          : Matches the 'z' suffix after the hour
      pattern = r'\.(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})\.t(?P<hour>\d{2})z'
    
      match = re.search(pattern, filename)
    
      if not match:
            raise ValueError(f"Could not parse reference time from filename: {filename}")

      # Convert captured strings to integers
      # Note: Adding 2000 to the 2-digit year (e.g., '25' -> 2025)
      year = int(match.group('year')) + 2000
      month = int(match.group('month'))
      day = int(match.group('day'))
      hour = int(match.group('hour'))
    
      # Create datetime object
      ref_dt = datetime(year, month, day, hour)
    
      # Return as numpy datetime64
      return np.datetime64(ref_dt)

    def _load_text_blacklist(self, file_path):
        """Parses the txt formatted blacklist into a dictionary cache."""
        # self.blacklist_data structure: { 'kt_type': {'sids': set(), 'kxs': set()} }
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    # Skip comments and empty lines [cite: 26, 27]
                    if line.startswith('!') or not line.strip() or line.startswith('['):
                        continue

                    parts = line.split()
                    if len(parts) >= 3:
                        kt = parts[0]      # e.g., 't', 'ps', 'q', 'uv'
                        kx = int(parts[1])  # e.g., 120, 220
                        sid = parts[2]     # e.g., 42101

                        if kt not in self.blacklist_data:
                            self.blacklist_data[kt] = {'sids': [], 'kxs': [] }

                        self.blacklist_data[kt]['sids'].append(sid)
                        self.blacklist_data[kt]['kxs'].append(kx)
        except Exception as e:
            self.log.error(f"Failed to load blacklist: {e}")


    def _get_blacklist(self, container: bufr.DataContainer, kt_type, catID=[]):
        """
        Apply GMAO blacklist for conventional observations by parsing the raw text file.
        Parameters: container object, kt type of observation (e.g., 't', 'q', 'ps'), file location of text blacklist
        """
         
        sid = container.get('stationIdentification', catID)
        typ = container.get('observationType', catID)
        clean_sids = np.char.strip(sid.astype(str))

        # 1. Check that blacklist data is present for KT type 
        if kt_type not in self.blacklist_data:
            return np.full_like(clean_sids, False, dtype=bool)
 
        # 2. Load blacklist data 
        blacklisted_sids = np.array(list(self.blacklist_data[kt_type]['sids']))
        blacklisted_kxs = np.array(list(self.blacklist_data[kt_type]['kxs']))
        reference_lookup=set(zip(blacklisted_sids,blacklisted_kxs))

        # 3. Create Masks
        mask = [ (s, i) in reference_lookup for s, i in zip(clean_sids, typ.astype(int)) ]

        return np.asarray(mask) 

    def _filter_identical(self, container: bufr.DataContainer,issurfacepressure='NA',catID=[]):
        """
        Removes observations with identical lat,lon,pressure,time and station ID
        To correspond to GSI setup scripts 

        Parameters: container object, boolean - is surface pressure ob (these do not compare pres)
        """

        # Example Data
        lat      = container.get('latitude',catID) 
        lon      = container.get('longitude',catID) 
        time     = container.get('timestamp',catID) 
        pressure = container.get('pressure',catID) 
        sid      = container.get('stationIdentification',catID) 
        otype    = container.get('observationType',catID)
        dtype = [('lat', 'f8'), ('lon', 'f8'), ('time', 'f8'), ('pres', 'f8'), ('sid', 'U10'),('otype', 'int64')]

        # 2. Create the structured array
        structured_data = np.empty(len(lat), dtype=dtype)
        structured_data['lat'] = lat.data
        structured_data['lon'] = lon.data
        structured_data['time'] = time.data
        structured_data['pres'] = pressure.data
        structured_data['sid'] = sid.data
        structured_data['otype'] = otype.data

        # 4. Find the first occurrences
        # We do NOT use axis=0 here. Because it's a structured array, 
        # each element is already a "row," so np.unique treats them as 1D.
        _, first_indices = np.unique(structured_data, return_index=True)
        # 5. Identify the duplicates (True where the sample should be masked)
        duplicate_mask = np.ones(lat.shape, dtype=bool)
        duplicate_mask[first_indices] = False

        # 6. Apply the mask to your MaskedArrays
        # Using |= ensures we keep any existing masks (e.g., NaNs or errors)
        for arr in [lat, lon, time, pressure, sid,otype]:
            arr.mask |= duplicate_mask
        return lat

    def _correct_drift_times(self, container: bufr.DataContainer, reference_time: np.datetime64,catID=[]) -> np.array:
        """
        where sonde data contains drift information, if high resolution time data indicates ob outside of cycle window
        then replace observation timestamp with sonde launch time 
        """
        times = container.get('obsTimeMinusCycleTime',catID)
        launch_times = container.get('launchTimeMinusCycleTime',catID) 
        typ = container.get('observationType',catID)
        timestamps = container.get('timestamp',catID) 
        dt_launch = ma.masked_array(np.round(3600 * launch_times).astype(np.int64),
                                      dtype='timedelta64[s]')
        drift_correction=(reference_time+dt_launch).astype('datetime64[s]').astype('int64')
        new_timestamps = ma.masked_array(np.where((np.isin(typ,self.driftdat_types) & (np.abs(times)>3) &\
                (~launch_times.mask)),drift_correction,timestamps),dtype='datetime64[s]',mask=timestamps.mask).astype('int64')
        container.replace('timestamp',new_timestamps,catID) 

    def _correct_surface_height(self, container: bufr.DataContainer,catID=[]) -> np.array:
        """
        Adds in the surface height corrections to ADPSFC and SFCSHP 
        subsets corresponding to corrections applied in GSI
        """
        typ=container.get('observationType',catID)
        t29=container.get('observationSubTypeNum',catID)
        selv=container.get('stationElevation',catID)
        height=container.get('height',catID)

        
        mask_ship = (typ==280)
        mask_atlas = (typ==282)
        mask_scatterometer=np.isin(typ, [285,289,290])
        mask_t29_ship = np.isin(t29, [522,523,531])
        height_s0 = selv + 10.0
        height_s1 = np.where((mask_ship) & (mask_t29_ship),20.0,height_s0)
        height_s2 = np.where(mask_atlas,selv+20.0,height_s1)
        height_s3 = np.where(mask_scatterometer, selv,height_s2)
        selv_s1=np.where(mask_scatterometer,0,selv)
        new_height=np.where(((typ<300)&(typ>=280))|((typ>=221)&(typ<=229)),height_s3,height)
        new_selv=np.where(((typ<300)&(typ>=280))|((typ>=221)&(typ<=229)),selv_s1,selv)

        container.replace('stationElevation',new_selv,catID)
        container.replace('height',new_height,catID)

    def _replace_timestamp(self, container: bufr.DataContainer, reference_time: np.datetime64, catID=[]) -> np.array:
        """
        fills timestamp field by adding 'obstime minus cycle time' 
        from bufr file to reference time pulled from filename
        """
        times = container.get('obsTimeMinusCycleTime',catID)

        cycle_times = ma.masked_array(np.round(3600 * times).astype(np.int64),
                                      dtype='timedelta64[s]',
                                      mask=times.mask)

        timestamps = ma.masked_array(reference_time + cycle_times,
                                     mask=times.mask,
                                     fill_value=bufr.get_missing_value(np.int64),
                                     dtype='datetime64[s]').astype('int64')

        container.replace('timestamp', timestamps,catID)
