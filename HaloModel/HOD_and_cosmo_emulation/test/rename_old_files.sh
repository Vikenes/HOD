#!/bin/bash

# root_dir="/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"
root_dir="/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit"

# Loop through directories AbacusSummit_base_c000_ph000-AbacusSummit_base_c000_ph024
# Make list of the three subdirectories we wish to loop over
# subdirectories=("HOD_catalogues" "HOD_parameters" "TPCF_data") 
# subdirectories=("HOD_catalogues") 


for i in {000..024}; do
    dir="$root_dir/AbacusSummit_base_c000_ph${i}/TPCF_data"
    # Check if directory exists, otherwise skip
    # for subdirectory in ${subdirectories[@]}; do
    if [ ! -d $dir ]; then
        continue
    fi
    ls -l $dir/*.hdf5
done

# for i in {001..140}; do
# for i in {140..181}; do
#     dir="$root_dir/AbacusSummit_base_c${i}_ph000/TPCF_data"
#     # Check if directory exists, otherwise skip
#     if [ ! -d $dir ]; then
#         continue
#     fi
#     ls -l $dir/*.hdf5
# done