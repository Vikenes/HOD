#!/bin/bash

root_dir="/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files"

# Loop through directories AbacusSummit_base_c000_ph000-AbacusSummit_base_c000_ph024
# Make list of the three subdirectories we wish to loop over
subdirectories=("HOD_catalogues") 
# subdirectories=("HOD_catalogues") 


# for i in {000..024}; do
#     # dir="$root_dir/AbacusSummit_base_c000_ph${i}"
#     dir="$root_dir/AbacusSummit_base_c000_ph${i}/HOD_catalogues"

#     # Check if directory exists, otherwise skip
#     echo $dir
#     ls -lh $dir/*
#     echo " " 
# done


# for i in {100..126}; do
#     # dir="$root_dir/AbacusSummit_base_c${i}_ph000"
#     dir="$root_dir/AbacusSummit_base_c${i}_ph000/HOD_catalogues"
#     # if [ ! -d $dir ]; then
#         # continue
#     # fi
#     # Check if directory exists, otherwise skip
#     echo $dir
#     ls -lh $dir/*
#     echo " " 
# done

# Loop over all directories beginning with AbacusSummit_base_c
for dir in $root_dir/AbacusSummit_base_c*; do
    # Check if directory exists, otherwise skip
    TPCF_dir="$dir/TPCF_data"
    # Check whether $dir exists, but not $TPCF_dir
    if [ ! -d $TPCF_dir ]; then
        echo "WARNING $TPCF_dir does not exist"
        break
    fi



    # Check if directory exists, otherwise skip
    # echo $dir
    # ls -lh $dir/*
    du -sh $TPCF_dir/.
    # echo " " 

    # Break loop
    # break
    # echo " " 
done
