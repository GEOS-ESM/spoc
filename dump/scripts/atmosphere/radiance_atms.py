#!/usr/bin/env python3
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('radiance_atms.yaml')
scan_interval = 8/3

class BufrAtmsObsBuilder(ObsBuilder):
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
                continue

            ifov = container.get('fieldOfViewNumber', cat)
            time = container.get('timestamp', cat)
            mintime = np.min(time)
            iscan = np.round((time - mintime) / scan_interval) + 1
            iscan = iscan.astype(int)

            mask_skip = (iscan % 5 == 0) & ((ifov - 3) % 5 == 0)
            container.apply_mask(mask_skip, cat)

            ifov = container.get('fieldOfViewNumber', cat)
            mask_edge = ((ifov >= 7) & (ifov <= 90))
            container.apply_mask(mask_edge, cat)

        # Check
        self.log.debug(f'container list (updated): {container.list()}')
        self.log.debug(f'all_sub_categories {container.all_sub_categories()}')

        return container


add_main_functions(BufrAtmsObsBuilder)
