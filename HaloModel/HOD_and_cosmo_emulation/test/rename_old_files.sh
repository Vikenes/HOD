#!/bin/bash

root_dir="/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"

# Loop through directories AbacusSummit_base_c000_ph000-AbacusSummit_base_c000_ph024
# Make list of the three subdirectories we wish to loop over
subdirectories=("HOD_catalogues" "HOD_parameters" "TPCF_data") 
# subdirectories=("HOD_catalogues") 


for i in {000..024}; do
    dir="$root_dir/AbacusSummit_base_c000_ph${i}"
    # Check if directory exists, otherwise skip
    for subdirectory in ${subdirectories[@]}; do
        if [ ! -d $dir/$subdirectory ]; then
            continue
        fi
        mv $dir/$subdirectory $dir/old_$subdirectory
    done
done

for i in {001..181}; do
    dir="$root_dir/AbacusSummit_base_c${i}_ph000"
    # Check if directory exists, otherwise skip
    if [ ! -d $dir ]; then
        continue
    fi
    for subdirectory in ${subdirectories[@]}; do
        if [ ! -d $dir/$subdirectory ]; then
            continue
        fi
        mv $dir/$subdirectory $dir/old_$subdirectory
    done
done