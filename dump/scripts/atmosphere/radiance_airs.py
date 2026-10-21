#!/usr/bin/env python3
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('radiance_airs.yaml')

BUFR_TABLE_PATH = os.getenv('BUFR_TABLE_PATH')
#print(f'BUFR_TABLE_PATH: {BUFR_TABLE_PATH}')


class BufrAIRSObsBuilder(ObsBuilder):

    def __init__(self):
        super().__init__(MAPPING_PATH, table_path=BUFR_TABLE_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        # Get container from mapping file
        self.log.info('Get container from bufr')
        container = super().make_obs(comm, input_path)

        self.log.debug(f'Container list (original): {container.list()}')
        self.log.debug(f'All_sub_categories =  {container.all_sub_categories()}')
        self.log.debug(f'Category map = {container.get_category_map()}')

        for cat in container.all_sub_categories():
            self.log.debug(f'category: {cat}')

            satId = container.get('satelliteId', cat)
            satId[:] = 49
            container.replace('satelliteId',satId,cat)

            if not np.any(satId):
                self.log.warning(f'Category {cat[0]} does not exist in input file')

        # Final container state
        self.log.debug(f'Container list (updated): {container.list()}')
        self.log.debug(f'Categry {container.all_sub_categories()}')

        return container

add_main_functions(BufrAIRSObsBuilder)
