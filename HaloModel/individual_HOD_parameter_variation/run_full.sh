subfolder='presentation'
N_samples=10


# Extreme test
sigma_logM=(0.01 1.0)
log10M1=(13.0 15.8)
kappa=(0.001 3.0)
alpha=(0.1 1.5)
params_array=(${sigma_logM[@]} ${log10M1[@]} ${kappa[@]} ${alpha[@]})

overwrite_existing_files=1 # Overwrite existing files? 1=yes, 0=no.


echo "Running full pipeline for subfolder '$subfolder' with $N_samples samples."
echo "Generating HOD parameters."
python3 generate_HOD_params_individual.py $subfolder $N_samples $overwrite_existing_files ${params_array[@]}
exit_status_generate_HOD_params=$?

if [ $exit_status_generate_HOD_params -ne 0 ]; then
    echo "'generate_HOD_params_individual.py' encountered an error. Stopping."
    exit 1
fi

echo " "
echo "Generating galaxy catalogues."
python3 make_HOD.py $subfolder $overwrite_existing_files 
exit_status_make_HOD=$?
if [ $exit_status_make_HOD -ne 0 ]; then
    echo "'make_HOD.py' encountered an error. Stopping."
    exit 1
fi

echo " "
echo "Computing TPCF's."
python3 compute_tpcf.py $subfolder $overwrite_existing_files 
exit_status_compute_tpcf=$?
if [ $exit_status_compute_tpcf -ne 0 ]; then
    echo "'compute_tpcf.py' encountered an error. Stopping."
    exit 1
fi

echo " "
echo "Plotting TPCF's."
python3 plot_tpcf.py $subfolder
