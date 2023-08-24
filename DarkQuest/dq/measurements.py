from pathlib import Path
import numpy as np
import pandas as pd
from dq import constants
from dq import snapshot_to_scale_factor
from dq import Cosmology
from dq import convert_run_to_cosmo_number

HALO_DATA_DIR = Path("/cosma6/data/dp004/dc-cues1/DarkQuest/")
HALO_VEL_DATA_DIR = Path("/cosma7/data/dp004/dc-cues1/DarkQuest/")
HOD_DATA_DIR = Path(
    "/cosma7/data/dp004/dc-cues1/DarkQuest/mock_catalogues/"
)


class Measured:
    def __init__(self,):
        self.planck_runs = np.arange(101, 115)

    def average_dataframes(self, dfs):
        df_concat = pd.concat(dfs)
        return df_concat.groupby(df_concat.index).mean()

    def std_dataframes(self, dfs):
        df_concat = pd.concat(dfs)
        return df_concat.groupby(df_concat.index).std()

    def get_planck(self, f, **kwargs):
        dfs = []
        for run in self.planck_runs:
            dfs.append(f(run=run, **kwargs))
        return self.average_dataframes(dfs)

    def get_planck_errors(self, f, **kwargs):
        dfs = []
        for run in self.planck_runs:
            dfs.append(f(run=run, **kwargs))
        return self.std_dataframes(dfs)#/np.sqrt(len(self.planck_runs))

    def get_lognh_bin(self, logn1: float, logn2: float) -> int:
        lognh_table = np.loadtxt(HALO_DATA_DIR / "xi/log10density_table.dat")
        idx = [
            i
            for i, (n1_, n2_) in enumerate(lognh_table)
            if (logn1 == n1_) and (logn2 == n2_)
        ]
        if idx:
            return idx[0]
        raise ValueError("n1 or n2 are not valid")

    def xi_hh(self, run: int, snapshot: int, logn1: float, logn2: float):
        r_xi = np.loadtxt(HALO_DATA_DIR / "xi/separation.dat")
        logn_bin = self.get_lognh_bin(logn1=logn1, logn2=logn2)
        if run > 100:
            xi_hh_data = np.load(HALO_DATA_DIR / "xi/xihh_fiducial.npy")
            xi = xi_hh_data[snapshot, logn_bin]
        else:
            xi_hh_data = np.load(HALO_DATA_DIR / "xi/xihh.npy")
            cnumber = convert_run_to_cosmo_number(run-1)
            xi = xi_hh_data[cnumber, snapshot, logn_bin]
        return pd.DataFrame({"r_c": r_xi, "xi": xi,})

    def planck_xi_hh(self, snapshot: int, logn1: float, logn2: float):
        return self.xi_hh(run=101, snapshot=snapshot, logn1=logn1, logn2=logn2)

    def planck_xi_error_hh(self, snapshot: int, logn1: float, logn2: float):
        r_xi = np.loadtxt(HALO_DATA_DIR / "xi/separation.dat")
        logn_bin = self.get_lognh_bin(logn1=logn1, logn2=logn2)
        xi_err = np.load(HALO_DATA_DIR / "xi/xihh_error_fiducial.npy")[
            snapshot, logn_bin
        ] * np.sqrt(len(self.planck_runs))
        return pd.DataFrame({"r_c": r_xi, "xi": xi_err,})

    def multipoles_hh(
        self, run: int, snapshot: int, logn1: float, logn2: float, los: int
    ) -> np.array:
        if logn1 != logn2:
            raise ValueError("Not measured!")
        return pd.read_csv(
            HALO_DATA_DIR
            / f"baorsd/tpcfs/xi_l_run{str(run).zfill(3)}_nd{abs(logn1):.2f}_s{str(snapshot).zfill(3)}_los{los}.csv"
        )

    def planck_multipoles_hh(self, snapshot: int, logn1: float, logn2: float):
        dfs = []
        for run in self.planck_runs:
            for los in [0, 1, 2]:
                dfs.append(
                    self.multipoles_hh(
                        run=run, snapshot=snapshot, logn1=logn1, logn2=logn2, los=los
                    )
                )
        return self.average_dataframes(dfs)

    def planck_multipoles_errors_hh(self, snapshot: int, logn1: float, logn2: float):
        dfs = []
        for run in self.planck_runs:
            for los in [0, 1, 2]:
                dfs.append(
                    self.multipoles_hh(
                        run=run, snapshot=snapshot, logn1=logn1, logn2=logn2, los=los
                    )
                )
        return self.std_dataframes(dfs)

    def n_g(self, run: int, snapshot: int, mcmc: bool = False):
        if mcmc:
            file_path = HOD_DATA_DIR / f'my_mocks/lowz/lr_nfw_diemer_R{run}_S{snapshot}_masscorr.csv'
        else:
            file_path = HOD_DATA_DIR / f'my_mocks/lr_nfw_R{run}_S{snapshot}_masscorr.csv'
        with open(file_path) as f:
            data = f.readlines()
            lines = len(data)
        return lines/constants.boxsize**3

    def planck_n_g(self, snapshot: int, mcmc: bool = False):
        n_g = []
        for run in self.planck_runs:
            n_g.append(self.n_g(run=run, snapshot=snapshot, mcmc=mcmc))
        return np.mean(n_g)

    def planck_error_n_g(self, snapshot: int, mcmc: bool = False):
        n_g = []
        for run in self.planck_runs:
            n_g.append(self.n_g(run=run, snapshot=snapshot, mcmc=mcmc))
        return np.std(n_g)

    def xi_gg(
        self, run: int, snapshot: int, galaxy_type: str = "g_g", mcmc: bool = False
    ):
        if mcmc:
            return pd.read_csv(
                HOD_DATA_DIR / f"summary_statistics_mcmc/xi_real/xi_run{run}_s{str(snapshot).zfill(3)}_lowz.csv"
            )

        return pd.read_csv(
                HOD_DATA_DIR / f"summary_statistics_mcmc/xi_real/xi_run{run}_s{str(snapshot).zfill(3)}_log_lowz.csv"
        )

    def planck_xi_gg(self, snapshot: int, galaxy_type: str = "g_g", mcmc: bool = False):
        return self.get_planck(
            self.xi_gg, snapshot=snapshot, galaxy_type=galaxy_type, mcmc=mcmc
        )

    def planck_error_xi_gg(
        self, snapshot: int, galaxy_type: str = "g_g", mcmc: bool = False
    ):
        return self.get_planck_errors(
            self.xi_gg, snapshot=snapshot, galaxy_type=galaxy_type, mcmc=mcmc
        )

    def multipoles_gg(self, run: int, snapshot: int, galaxy_type: str = "g_g"):
        # TODO: Add los
        return pd.read_csv(
            HOD_DATA_DIR
            / f"my_xi_s/{galaxy_type}_xi_l_run{run}_s{str(snapshot).zfill(3)}_los0.csv"
        )

    def planck_multipoles_gg(self, snapshot: int, galaxy_type: str = "g_g"):
        return self.get_planck(
            self.multipoles_gg, snapshot=snapshot, galaxy_type=galaxy_type
        )

    def planck_error_multipoles_gg(self, snapshot: int, galaxy_type: str = "g_g"):
        return self.get_planck_errors(
            self.multipoles_gg, snapshot=snapshot, galaxy_type=galaxy_type
        )

    def get_velocity_moments(self, velocity_path) -> np.ndarray:
        if velocity_path.is_file():
            return pd.read_csv(
                velocity_path,
                sep="\t",
                skiprows=1,
                names=(
                    "r",
                    "v_r",
                    "sigma_r",
                    "sigma_t",
                    "skew_r",
                    "skew_rt",
                    "kurtosis_r",
                    "kurtosis_t",
                    "kurtosis_rt",
                ),
            )

        else:
            raise ValueError

    def convert_moments(self, df, cosmology, a):
        kms_to_Mpc = cosmology.km_s_to_Mpc_h(a=a)
        df["v_r"] *= kms_to_Mpc
        df["sigma_r"] *= kms_to_Mpc
        df["sigma_t"] *= kms_to_Mpc
        return df

    def moments_hh(
        self, run: int, snapshot: int, logn1: float, logn2: float,
    ):
        cosmology = Cosmology.from_run(run=run)
        a = snapshot_to_scale_factor(snapshot=snapshot)
        velocity_path = (
            HALO_VEL_DATA_DIR
            / f"pairwise_velocities/run{run}_nd_{abs(logn1):.1f}_{abs(logn2):.1f}_snapshot{snapshot}.csv"
        )
        df = self.get_velocity_moments(velocity_path)
        return self.convert_moments(df=df, cosmology=cosmology, a=a)

    def planck_moments_hh(self, snapshot: int, logn1: float, logn2: float):
        return self.get_planck(
            self.moments_hh, snapshot=snapshot, logn1=logn1, logn2=logn2,
        )

    def planck_errors_moments_hh(self, snapshot: int, logn1: float, logn2: float):
        return self.get_planck_errors(
            self.moments_hh, snapshot=snapshot, logn1=logn1, logn2=logn2,
        )

    def moments_gg(
        self, run: int, snapshot: int, galaxy_type: str = "g_g",
    ):
        cosmology = Cosmology.from_run(run=run)
        a = snapshot_to_scale_factor(snapshot=snapshot)
        velocity_path = (
            HOD_DATA_DIR
            / f"summary_statistics/my_pairwise_velocities/{galaxy_type}_v_run{run}_snapshot{snapshot}.csv"
        )
        df = self.get_velocity_moments(velocity_path)
        return self.convert_moments(df=df, cosmology=cosmology, a=a)

    def planck_moments_gg(self, snapshot: int, galaxy_type: str = "g_g"):
        return self.get_planck(
            self.moments_gg, snapshot=snapshot, galaxy_type=galaxy_type
        )

    def all_xi_hh(
        self,
        snapshot: int,
        min_logn: float,
        max_logn: float,
        run: int=101,
        r_max: float = 60.0,
    ):

        logdens = np.arange(max_logn, min_logn-0.5, -0.5)
        df = self.planck_xi_hh(snapshot=snapshot, logn1=logdens[0], logn2=logdens[0])
        r = df['r_c'].values
        n_lower_diag = len(logdens) * (len(logdens)+1) // 2
        xi_hh = np.zeros((n_lower_diag, len(r[r<r_max])))
        counter = 0
        for i, n1 in enumerate(logdens):
            for j, n2 in enumerate(logdens):
                if n1 >= n2:
                    if run > 100:
                        df = self.planck_xi_hh(snapshot=snapshot, logn1=n1, logn2=n2)
                    else:
                        df = self.xi_hh(run=run, snapshot=snapshot, logn1=n1, logn2=n2)
                    xi = df['xi'].values
                    xi_hh[counter, :] = xi[r < r_max]
                    counter += 1
        return r[r < r_max], xi_hh


