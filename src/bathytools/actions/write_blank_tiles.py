import logging
import string as sg

import MITgcmutils as mit
import numpy as np

from bathytools.actions import SimpleAction
from bathytools.output_appendix import OutputAppendix


LOGGER = logging.getLogger(__name__)

TEMPLATE = """
 &W2_EXCH2_PARM01
 W2_mapIO   = 1,
 preDefTopol=1,
 
 dimsFacets = $nx, $ny,
 blankList = $blanks
 
### number of water tiles ###
### N_water_tiles = $nwater
### number of land tiles ###
### N_land_tiles = $nland
### percentage of water tiles ###
### water_fraction = $wfrac
 &"""


class WriteBlankTiles(SimpleAction):
    """ """

    def __init__(
        self,
        name: str,
        description: str,
        output_appendix: OutputAppendix,
        file_name: str,
        size_tiles_lon: int,
        size_tiles_lat: int,
    ):
        super().__init__(name, description, output_appendix=output_appendix)

        self._output_file = output_appendix.output_dir / file_name
        self._size_tiles_lon = size_tiles_lon
        self._size_tiles_lat = size_tiles_lat

    def __call__(self, bathymetry):
        tmplt = sg.Template(TEMPLATE)

        lat_size = self._size_tiles_lat
        lon_size = self._size_tiles_lon

        depth = -bathymetry.elevation.transpose("latitude", "longitude").values
        list_tiles = [
            int(k) for k in mit.utils.gen_blanklist(depth, lon_size, lat_size)
        ]
        ny, nx = depth.shape
        nland = len(list_tiles)
        n_proc_y_untiled, n_proc_x_untiled = ny//lat_size, nx//lon_size
        nwater = n_proc_y_untiled * n_proc_x_untiled - nland
        nnewline = int(np.ceil(len(list_tiles) / 10))
        lnl = [list_tiles[10 * i : 10 * (i + 1)] for i in range(nnewline)]
        snl = [str(lnli)[1:-1] + "," for lnli in lnl]

        blank_dict = {"blanks": "\n\t".join(snl), "nx": str(nx), "ny": str(ny), "nland": str(nland), "nwater": str(nwater), "wfrac": f"{100*nwater/(nwater+nland):.1f}"}

        LOGGER.info("Writing MIT tile template to %s", self._output_file)
        with self._output_file.open("w") as f:
            f.write(tmplt.substitute(blank_dict))

        blanks_file = self._output_appendix.output_dir / "blanks.txt"
        LOGGER.info("Writing blank tiles list into %s", blanks_file)

        np.savetxt(blanks_file, np.asarray(list_tiles, dtype=int), fmt="%d")

        return bathymetry
