# HaloModel and DarkQuest workings 
Here we describe how the two packages `HaloModel` and `DarkQuest` works in practice to create the galaxy catalogues. 
This note was mainly written for myself, and was intended to serve as a reference for myself on which parts of the pipeline were important.   

The main pipeline is as follows:
 - Define cosmology: `dq.Cosmology`
 - Initialize halo catalogue: `hmd..HaloCatalogue`
    - Compute halo radius (Uses cosmology)
    - Compute velocity dispersion (Uses cosmology)
    - Set up concentration: `hmd.concentraion.diemer15`
 - Initialize HOD maker: `hmd.populate.HODMaker`
 - Call():
    - Get central: `Zheng07Centrals`
    - Get satellite: `Zheng07Sats`
    - Add profile (satellites): 
        - sample positions from profile 
        - sample vels from profile 
    - Create galaxy DataFrame
        - Consists of central+satellites
        - Apply PBC
    - Initialize GalaxyCatalogue.from_frame()
        - Not directly used 


#### dq - Cosmology
Uses the class `Cosmology(wCDM)`, where `wCDM` is the `astropy.cosmology.wCDM` class: "*FLRW cosmology with a constant dark energy EoS and curvature.*". Initializes all cosmological parameters, and is used to compute critical density, density parameters, etc. 

### hmd.catalogue.HaloCatalogue
Parameters (With N halos):
 - `pos`: shape (N, 3)
 - `vel`: shape (N, 3)
 - `mass`: shape (N, )
 - `boxsize`: float 2GPc/h
 - `conc_mass_model`: hmd.concentration.diemer15
 - `cosmology`: dq.Cosmology(wCDM)
 - `redshift`: z=0.25 

When initialized, it computes:
 - **halo radius:** 
    - `halotools.empirical_models.halo_mass_to_halo_radius`:
        - `mass`: (N, ) 
        - `cosmology`
        - `redshift`
        - `mdef=200m`
    - mdef=200m means that $\rho_\mathrm{threshold}(z)=200\rho_m(z)$
    - $\rho_m=\Omega_m(z) \rho_\mathrm{crit}$, where $\Omega_m(z)$ is gotten from cosmology.Om(redshift) 
    - Uses the relation $M_{\Delta}(z) \equiv \frac{4\pi}{3}R_{\Delta}^{3}\Delta_{\rm ref}(z)\rho_{\rm ref}(z)$ to compute the radius, $R_\Delta (z)$.
 - **velocity dispersion:**
    - vel_disp=vel_virial
    - `halotools.empirical_models.halo_mass_to_virial_velocity`:
        - `mass`: (N, ) 
        - `cosmology`
        - `redshift`
        - `mdef=200m`
    - $V_{\rm vir} \equiv \sqrt{GM_{\rm halo}/R_{\rm halo}}$, get radius from mass. 
    - Weird factor in the end, I don't understand it.
 - **Concentration:**
    - `hmd.concentration.diemer15`:
        - `prim_haloprop=mass`
        - `cosmology`
        - `redshift`
        - `sigma8=get_sigma8` (using dark-emulator package)
        - `allow_tabulation=True`
    - Computes c from Eq. (9) in [Diemer & Kravtsov (2015)](https://iopscience.iop.org/article/10.1088/0004-637X/799/1/108/pdf), using updated parameters from Diemer & Joyce (2019)
    - The halo concentration computed here is the concentration used by central galaxies. 
    - The same concentration is inherited by the satellite galaxies as well  


### hmd.populate.HODMaker
Parameters 
 - `halo_catalogue`: The catalogue generated with `hmd.catalogue`
 - `central_occ=Zheng07Centrals()`
 - `sat_occ=Zheng07Sats()`
 - `satellite_profile=FixedCosmologyNFW`
    - `cosmology=halocat.cosmology`
    - `redshift`
    - `mdef=200m`
    - `conc_mass_model=dutton_maccio14`
    - `sigmaM=None`
 - `galaxy=Galaxy(HOD_params)`

When initialized it does the following:
 - `halo_catalogue.to_frame()`: Make halo catalogue into a `pandas` DataFrame 

 When called, having N halos:
 - `get_central_df()`:
    - `probability=central_occ.get_n(halo_mass,halo_rank,galaxy)`: Returns $\langle N_c \rangle$
    - Set up `randoms` array of size (N,) in range $[0,1)$
    - `central_df=halo_df[probability>randoms]`
    - Computes vel_disp: Not used 
 - `get_satellite_df(central_df,)`:
    - `lambda_sat=sat_occ.lambda_sat(M)`: Returns $\lambda_s\neq 0$ for $M_h>\kappa M_\mathrm{min}$
    - `n_sats=np.random.poisson(lambda_sat)`: Get the number of satellite galaxies in a halo according to the Poisson distribution 
    - `satellite_df=central_df.loc[central_df.index.repeat(n_sats)]`: Populate the central satellites with `n_sat` satellite galaxies, by repeating entries of central df.
    - `satellite_df["host_id"]=satellite_df.index`: Assign same halo host ID to satellites occupying he same host halo. 
    - Position satellite galaxies in 3D space with `satellite_df = self.add_profile(satellite_df=satellite_df)`
        - `(halo_centric_r,x,y,z)=sample_positions_from_profile(sat_df["conc"], n_sats=len(sat_df))`
            - Follow Eq. (6) in [Robotham & Howlett (2018)](https://arxiv.org/pdf/1805.09550.pdf)
            - Assume NFW profile, and that satellite galaxies trace DM. Scale radii according to inverted CDF.  
