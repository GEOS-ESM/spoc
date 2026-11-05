#!/usr/bin/env python3
import json
import netCDF4 as nc
import os

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

SPC_COEFF_VERSION = 1
INVALID = 1000.0
NMFD = '1b'   # Normal Feed
RARS = 'es'   # Regional ATOVS Retransmission Services

# Cosmic background temperature. Taken from Mather,J.C. et. al., 1999, "Calibrator Design for the COBE
# Far-Infrared Absolute Spectrophotometer (FIRAS)"Astrophysical Journal, vol 512, pp 511-520
COSMIC_BACKGROUND_TEMP = 2.7253

nc_dir = './'


class ACCoeff:
    """
    Loads and manages AMSU-A antenna correction coefficients for a specified satellite.

    This class reads correction coefficients from a NetCDF file corresponding to the given
    satellite ID. The coefficients are used for antenna correction in satellite brightness
    temperature processing.

    Attributes:
        n_fovs (int): Number of fields of view (FOVs) in the instrument.
        n_channels (int): Number of instrument channels.
        a_earth (ndarray): Earth-view antenna correction coefficients.
        a_platform (ndarray): Platform-view antenna correction coefficients.
        a_space (ndarray): Space-view antenna correction coefficients.
        a_ep (ndarray): Combined Earth and platform correction coefficients.
        a_sp (ndarray): Space correction coefficients scaled by the cosmic background temperature.

    Args:
        ac_dir (str): Directory containing ACCoeff NetCDF files.
        sat_id (str, optional): Satellite ID string (default 'n19').

    Example:
        ac = ACCoeff('/path/to/ac/files', sat_id='n19')
        print(ac.a_ep.shape)
    """
    def __init__(self, ac_dir, instrument='amsua', sat_id='n19'):
        file_name = os.path.join(ac_dir, instrument + '_' + sat_id + '.ACCoeff.nc')
        nc_file = nc.Dataset(file_name)
        self.file_name = file_name
        self.n_fovs = len(nc_file.dimensions['n_FOVs'])
        self.n_channels = len(nc_file.dimensions['n_Channels'])
        self.a_earth = nc_file.variables['A_earth'][:]
        self.a_platform = nc_file.variables['A_platform'][:]
        self.a_space = nc_file.variables['A_space'][:]
        self.a_ep = self.a_earth + self.a_platform
        self.a_sp = self.a_space * COSMIC_BACKGROUND_TEMP


class AtovsObsBuilder(ObsBuilder):
    """
    ObsBuilder subclass for Atovs satellite data.

    Handles mapping, parsing, correction, and merging of AMSU-A 1B and ESA data
    using their respective mapping files.
    """

    def __init__(self, map_dict, log_name, instrument='amsua'):
        """
        Initialize the AmsuaObsBuilder.

        Sets up mapping dictionaries for 1B and ESA data types.
        """

        super().__init__(map_dict, log_name=log_name)
        self.instrument = instrument

    def make_obs(self, comm, input_path):
        """
        Create observation container by parsing and merging 1B and ESA files.

        Parameters
        ----------
        comm : object
            MPI communicator.
        input_path : dict or str
            Dictionary (or JSON string) with keys '1b' and 'es' specifying file paths.

        Returns
        -------
        container : object
            Combined data container with merged and remapped variables.

        Raises
        ------
        ValueError
            If input_path is not a dictionary with the required keys.
        """

        if isinstance(input_path, str):
            input_path = json.loads(input_path)
        if not (isinstance(input_path, dict) and len(self.map_dict) <= 2):
            raise ValueError('The input must be a dict with one or two items!')

        self.log.info(f'input files: {input_path}')
        self.log.info(f'maping files: {self.map_dict["es"]}, {self.map_dict["1b"]}')
        nmfd_flag = False
        rars_flag = False
        total_files = 0
        if input_path.get('es'):
            container_es = bufr.Parser(input_path[RARS], self.map_dict[RARS]).parse(comm)
            self._re_map_variable(container_es, feed_type=RARS)
            rars_flag = True
            total_files += 1
        if input_path.get('1b'):
            self.log.info('Processing 1b file.')
            container_1b = bufr.Parser(input_path[NMFD], self.map_dict[NMFD]).parse(comm)
            self._re_map_variable(container_1b, feed_type=NMFD)
            nmfd_flag = True
            total_files += 1

        if total_files == 2:
            container = container_1b
            container.append(container_es)
        elif total_files == 1:
            if rars_flag:
                container = container_es
            else:
                container = container_1b
        return container

    def _remove_ant_corr(self, i, ac, ifov, t):
        """
        Remove antenna correction from brightness temperature.

        Parameters
        ----------
        i : int
            Index of the field of view.
        ac : array-like
            Antenna correction coefficients.
        ifov : array-like
            Field of view numbers.
        t : array-like
            Brightness temperature array.

        Returns
        -------
        array-like
            Brightness temperature with antenna correction removed.
        """

        t = ac.a_ep[i, ifov] * t + ac.a_sp[i, ifov]
        t[(ifov < 0) | (ifov >= ac.n_fovs)] = INVALID
        return t

    def _apply_ant_corr(self, i, ac, ifov, t):
        # t:              on input, this argument contains the antenna temperatures for the sensor channels.
        t = (t - ac.a_sp[i, ifov]) / ac.a_ep[i, ifov]
        t[(ifov < 0) | (ifov >= ac.n_fovs)] = INVALID
        return t

    def _apply_corr(self, sat_id, ta, ifov, feed_type=NMFD, sacv=None):

        ifov = ifov.astype(int) - 1
        ac = ACCoeff(nc_dir, instrument=self.instrument, sat_id=sat_id)
        self.log.info(f'ac file name: {ac.file_name}.')
        # Convert antenna temperature to brightness temperature
        for i in range(ta.shape[1]):
            self.log.debug(f'inside loop for allpy ta to tb: i = {i}')
            x = ta[:, i]
            if feed_type == RARS:
                self.log.debug('processing RARS')
                x[sacv] = self._remove_ant_corr(i, ac, ifov[sacv], x[sacv])
                x[sacv] = self._apply_ant_corr(i, ac, ifov[sacv], x[sacv])
            else:
                x = self._apply_ant_corr(i, ac, ifov, x)
            x[x >= INVALID] = INVALID
            ta[:, i] = x
        return ta

    def _re_map_variable(self, container, feed_type=NMFD):
        """
        Remap variables in the data container after parsing.

        Parameters
        ----------
        container : object
            Data container to be remapped.

        Returns
        -------
        None
        """

        sat_ids = container.all_sub_categories()
        for sat_id in sat_ids:
            self.log.info(f'Converting for {sat_id}, ...')
            sacv_flag = None
            if feed_type == NMFD and sat_id in ['n15', 'n16']:
                continue
            ta = container.get('brightnessTemperature', sat_id)
            self.log.info(f'This file has {ta.shape[0]} records.')
            if ta.shape[0]:
                if feed_type == RARS:
                    sacv = container.get('sacv', sat_id)
                    sacv_flag = (sacv != SPC_COEFF_VERSION)
                    if not sacv_flag.any():
                        continue

                ifov = container.get('fieldOfViewNumber', sat_id)
                tb = self._apply_corr(sat_id[0], ta, ifov, feed_type=feed_type, sacv=sacv_flag)
                container.replace('brightnessTemperature', tb, sat_id)
