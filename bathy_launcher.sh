#!/bin/bash

#configs="got gsn ion lig nad sad sar sic tyr"
configs="ion_0528 ion_0704 ion_1419 lig_0350 lig_0800 lig_1400 nad_0560 nad_0800 nad_1120 sad_0527 sad_0800 sad_1240 sar_0540 sar_0900 sar_1280 sic_0414 sic_0800 sic_1440 tyr_0408 tyr_0800 tyr_1632"
#configs="nad_560 nad_800 nad_1120"
names="ION LIG NAD SAD SAR SIC TYR"
#names="GOT GSN ION LIG NAD SAD SAR SIC TYR"

for c in ${configs}; do
	echo $c
	#mkdir ${c::3}
	#mkdir ${c::3}/${c}
	#mv ${c}* ${c::3}/${c}
	#poetry run bathytools --config ${c::3}/${c}/${c}.yaml -o ${c::3}/${c}/
	rm ${c::3}/${c}/bathy.bin ${c::3}/${c}/hFacC.bin ${c::3}/${c}/meshmask.nc ${c::3}/${c}/MIT_static.nc ${c::3}/${c}/rivers*
	echo " Done..."
done
