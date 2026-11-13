#!/usr/bin/env python3
import os
import numpy as np
from bufr.obs_builder import add_main_functions, map_path

from radiance_atovs_obs_builder import AtovsObsBuilder, NMFD, RARS


NMFD_MAPPING = map_path('radiance_amsua_1bamua.yaml')
RARS_MAPPING = map_path('radiance_amsua_esamua.yaml')


class AmsuaObsBuilder(AtovsObsBuilder):
    """
    ObsBuilder subclass for AMSU-A satellite data.

    Handles mapping, parsing, correction, and merging of AMSU-A 1B and ESA data
    using their respective mapping files.
    """

    def __init__(self):
        """
        Initialize the AmsuaObsBuilder.

        Sets up mapping dictionaries for 1B and ESA data types.
        """

        map_dict = {NMFD: NMFD_MAPPING,
                    RARS: RARS_MAPPING}

        super().__init__(map_dict, log_name=os.path.basename(__file__))
        
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
            mask = ((ifov >= 4) & (ifov <=27))
            container.apply_mask(mask, cat)

        return container
                                                         

add_main_functions(AmsuaObsBuilder)
