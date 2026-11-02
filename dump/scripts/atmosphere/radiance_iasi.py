#!/usr/bin/env python3
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


MAPPING_PATH = map_path('radiance_iasi.yaml')


class BufrIasiObsBuilder(ObsBuilder):
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

            self.log.debug(f'category = {cat}')

            satId = container.get('satelliteId', cat)
            if not np.any(satId):
                self.log.warning(f'category {cat[0]} does not exist in input file')

            # remove sensorScanPosition from 1 to 4 and from 57 to 60
            ifov = container.get('sensorScanPosition', cat)
            mask_edge = ((ifov >= 5) & (ifov <= 56))
            container.apply_mask(mask_edge, cat)

            # remove non-zero data quality Flag
            qFlags = container.get('gqisFlagQual', cat)
            mask_edge = (qFlags == 0)
            container.apply_mask(mask_edge, cat)

            # remove scanLineNumber with value of -1
            sLNumber = container.get('scanLineNumber', cat)
            mask_edge = (sLNumber != -1)
            container.apply_mask(mask_edge, cat)

        # Check
        self.log.debug(f'container list (updated): {container.list()}')
        self.log.debug(f'all_sub_categories {container.all_sub_categories()}')

        return container


add_main_functions(BufrIasiObsBuilder)
