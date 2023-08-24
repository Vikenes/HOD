import numpy as np 
import h5py 
from dq import Cosmology
import hmd
from hmd.catalogue import ParticleCatalogue, HaloCatalogue, GalaxyCatalogue
from hmd.occupation import Zheng07Centrals, Zheng07Sats
from hmd.galaxy import Galaxy
from hmd.profiles import FixedCosmologyNFW
from hmd.populate import HODMaker


file_halocat = h5py.File(
    f"/mn/stornext/d8/data/chengzor/MGGLAMx100/GR_halocat_z0.25.hdf5", 
    'r',
)
boxsize = 1024.0
redshift = 0.25
box_index = 1
pos = file_halocat[f"box{box_index}"]["halo_pos"][...] # shape: (N_halos, 3)
vel = file_halocat[f"box{box_index}"]["halo_vel"][...]
mass = file_halocat[f"box{box_index}"]["halo_mass"][...]
cosmology = Cosmology.from_run(101)
# cosmology.H0 = 70.0  
# H0, Om0, Ode0, w0, wc0, Ob0, n_s


halocat = HaloCatalogue(
    pos,
    vel,
    mass,
    boxsize,
    conc_mass_model=hmd.concentration.diemer15,
    cosmology=cosmology,
    redshift=redshift,
    # mdef: str = "200m",
)


maker = HODMaker(
    halo_catalogue=halocat,
    central_occ=Zheng07Centrals(),
    sat_occ=Zheng07Sats(),
    satellite_profile=FixedCosmologyNFW(
        cosmology=halocat.cosmology,
        redshift=redshift,
        mdef="200m",
        conc_mass_model="dutton_maccio14",
        sigmaM=None,
    ),
    galaxy=Galaxy(
        logM_min = 13.62,
        sigma_logM = 0.6915,
        kappa = 0.51,
        logM1 = 14.42,
        alpha = 0.9168,
        M_cut = 10 ** 12.26,
        M_sat = 10 ** 14.87,
        concentration_bias = 1.0,
        v_bias_centrals = 0.0,
        v_bias_satellites = 1.0,
        B_cen = 0.,
        B_sat = 0.,
    ),
)
maker()
maker.galaxy_df.to_csv(
    f"./HOD.csv",
)
