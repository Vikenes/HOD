import numpy as np
import pandas as pd

from dq import Cosmology, PlanckCosmology, snapshot_to_redshift
from dq import convert_run_to_cosmo_number
from dq.measurements import HALO_DATA_DIR, Measured


def create_df(measured, runs, divide_by_bias=True):
    lognh = np.loadtxt(HALO_DATA_DIR / "xi/log10density_table.dat")
    # TODO: change this
    snapshots = [20] #np.arange(21)
    (
        run_indices,
        rs,
        wbs,
        wcs,
        Ols,
        lnAss,
        n_ss,
        ws,
        logn1s,
        logn2s,
        redshifts,
        biases,
        xis,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [], [])
    planck_cosmology = PlanckCosmology()
    for run in runs:
        print(f'Run = ', run)
        cosmo_number = convert_run_to_cosmo_number(run=run)
        cosmology = Cosmology.from_run(run=run)
        wb, wc, Ol, lnAs, n_s, w = cosmology.cparams
        for snapshot in snapshots:
            redshift = snapshot_to_redshift(snapshot=snapshot)
            for logn_bin, (logn1, logn2) in enumerate(lognh):
                xi_df = measured.xi_hh(
                    run=cosmo_number-1, snapshot=snapshot, logn1=logn1, logn2=logn2,
                )
                b1 = planck_cosmology.get_bias(snapshot=snapshot, logn=logn1,)
                b2 = planck_cosmology.get_bias(snapshot=snapshot, logn=logn2,)
                n = len(xi_df["r_c"])
                if divide_by_bias:
                    xi_hh = (xi_df["xi"].values) / b1 / b2
                else:
                    xi_hh = (xi_df["xi"].values) 
                rs += list(xi_df["r_c"])
                wbs += [wb] * n
                wcs += [wc] * n
                Ols += [Ol] * n
                lnAss += [lnAs] * n
                n_ss += [n_s] * n
                ws += [w] * n
                logn1s += [logn1] * n
                logn2s += [logn2] * n
                redshifts += [redshift] * n
                xis += list(xi_hh)
                biases += [b1 * b2] * n
                run_indices += [run] * n
    return pd.DataFrame(
        {
            "run": run_indices,
            "b1b2": biases,
            "logn1": logn1s,
            "logn2": logn2s,
            "wb": wbs,
            "wc": wcs,
            "Ol": Ols,
            "lnAs": lnAss,
            "ns": n_ss,
            "w": ws,
            "redshift": redshifts,
            "r": rs,
            "xi_hh": xis,
        }
    )

def convert_df_to_multi_df(df):
    # Map r values to integers
    print(df[(df['logn1'] == df['logn2']) & (df['logn1'] == -2.5)])
    r_values = df['r'].unique()
    mapping_dict = dict(zip(r_values, range(len(r_values))))
    df['r_int'] = df['r'].map(mapping_dict)
    xi_0 = df[df['r_int'] == 0]
    xi_20 = df[df['r_int'] == 10]
    print(xi_0[(xi_0['logn1'] == xi_0['logn2']) & (xi_0['logn1'] == -2.5)])
    print(xi_20[(xi_0['logn1'] == xi_0['logn2']) & (xi_0['logn1'] == -2.5)])
    # convert r - xi columns into xi_0, xi_1
    return multi_df

if __name__ == "__main__":
    OUTPUT_DIR = "/cosma7/data/dp004/dc-cues1/DarkQuest/datasets/"
    multi_output = True
    n_runs = 99 # is this right?
    val_runs = list(range(41, 51))
    test_runs = list(range(51, 52)) + [101]
    measured = Measured()
    test_df = create_df(measured, test_runs,)
    #val_df = create_df(measured, val_runs,)
    #train_df = create_df(
    #    measured, [run for run in range(1,n_runs+1) if run not in test_runs + val_runs],
    #)
    if multi_output:
        test_df = convert_df_to_multi_df(test_df)
        val_df = convert_df_to_multi_df(val_df)
        train_df = convert_df_to_multi_df(train_df)

        test_df.to_csv(OUTPUT_DIR + "multioutput_test.csv", index=False)
        val_df.to_csv(OUTPUT_DIR + "multioutput_val.csv")
        train_df.to_csv(OUTPUT_DIR + "multioutput_train.csv")
    else:
        test_df.to_csv(OUTPUT_DIR + "test.csv", index=False)
        val_df.to_csv(OUTPUT_DIR + "val.csv")
        train_df.to_csv(OUTPUT_DIR + "train.csv")

    assert set(test_df["run"].unique()) == set(test_runs)
    assert set(val_df["run"].unique()) == set(val_runs)

    assert any(run in val_df["run"].unique() for run in train_df["run"].unique())
    assert any(run in test_df["run"].unique() for run in train_df["run"].unique())
    #TODO: make sure all runs are used
