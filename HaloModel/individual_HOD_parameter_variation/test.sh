subfolder='articlepriors'

# sigma_logMmin=0.1
# sigma_logMmax=1.0
# log10M1min=13.0
# log10M1max=15.8
# kappamin=0.01
# kappamax=3.0
# alphamin=0.5
# alphamax=1.0


sigma_logM=(0.6915) 
log10M1=(14.42) # h^-1 Msun
kappa=(0.51) 
alpha=(0.9168)  
factors=(0.9 1.1)

# sigma_logM=(0.1 1.0)
# log10M1=(13.0 15.8)
# kappa=(0.01 3.0)
# alpha=(0.5 1.0)
fiducial=(0.6915 14.42 0.51 0.9168)
params_array=()


# params_array=(${sigma_logM[@]} ${log10M1[@]} ${kappa[@]} ${alpha[@]})

N_samples=15
echo "Running full pipeline for subfolder '$subfolder' with $N_samples samples."

echo "Generating HOD parameters."
# python3 generate_HOD_params_individual.py $subfolder $N_samples ${params_array[@]}

# echo ${sigma_logM[@]}
let s=2.1+0.9 | bc
echo $s