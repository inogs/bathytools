import logging
import string as sg

import MITgcmutils as mit
import numpy as np

from bathytools.actions import SimpleAction
from bathytools.output_appendix import OutputAppendix


LOGGER = logging.getLogger(__name__)

TEMPLATE = """
# EXCH2 Package: Wrapper-2 User Choice
#--------------------
#  preDefTopol   :: pre-defined Topology selector:
#                :: = 0 : topology defined from processing "data.exch2";
#                :: = 1 : simple, single facet topology;
#                :: = 2 : customized topology (w2_set_myown_facets)
#                :: = 3 : 6-facet Cube (3 face-dims: nRed, nGreen, nBlue).
#  dimsFacets    :: facet pair of dimensions (n1x,n1y, n2x,n2y ...)
#  facetEdgeLink :: Face-Edge connectivity map:
#    facetEdgeLink(i,j)=XX.1 : face(j)-edge(i) (i=1,2,3,4 <==> N,S,E,W)
#    is connected to Northern edge of face "XX" ; similarly,
#    = XX.2 : to Southern.E, XX.3 = Eastern.E, XX.4 = Western.E of face "XX"
#  blankList     :: List of "blank" tiles
#  W2_mapIO      :: global map IO selector (-1 = old type ; 0 = 1 long line in X
#                :: 1 = compact, mostly in Y dir)
#  W2_printMsg   :: option for information messages printing
#                :: <0 : write to log file ; =0 : minimum print ;
#                :: =1 : no duplicated print ; =2 : all processes do print
#--------------------
 &W2_EXCH2_PARM01
# W2_printMsg= 1,
 W2_mapIO   = 1,
  preDefTopol=1,
#-- 5 facets llc_120 topology (drop facet 6 and its connection):
  dimsFacets = $nx, $ny,
  blankList = $blanks
#-- full 6 facets llc_120 topology (equivalent to default preDefTopol=3):
# dimsFacets = 120, 360, 120, 360, 120, 120, 360, 120, 360, 120, 120, 120,
# facetEdgeLink(1:4,1)= 3.4, 6.1, 2.4, 5.1,
# facetEdgeLink(1:4,2)= 3.2, 6.3, 4.2, 1.3,
# facetEdgeLink(1:4,3)= 5.4, 2.1, 4.4, 1.1,
# facetEdgeLink(1:4,4)= 5.2, 2.3, 6.2, 3.3,
# facetEdgeLink(1:4,5)= 1.4, 4.1, 6.4, 3.1,
# facetEdgeLink(1:4,6)= 1.2, 4.3, 2.2, 5.3,
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

        depth = -bathymetry.elevation.values
        list_tiles = [
            int(k) for k in mit.utils.gen_blanklist(depth, lat_size, lon_size)
        ]
        ny, nx = bathymetry.transpose("latitude", "longitude").shape
        nnewline = round(len(list_tiles) / 10)
        lnl = [list_tiles[10 * i : 10 * (i + 1)] for i in range(nnewline)]
        snl = [str(lnli)[1:-1] + "," for lnli in lnl]

        blank_dict = {"blanks": "\n\t".join(snl), "nx": str(nx), "ny": str(ny)}

        LOGGER.info("Writing MIT tile template to %s", self._output_file)
        with self._output_file.open("w") as f:
            f.write(tmplt.substitute(blank_dict))

        blanks_file = self._output_appendix.output_dir / "blanks.txt"
        LOGGER.info("Writing blank tiles list into %s", blanks_file)

        np.savetxt(blanks_file, np.asarray(list_tiles, dtype=int), fmt="%d")

        return bathymetry
