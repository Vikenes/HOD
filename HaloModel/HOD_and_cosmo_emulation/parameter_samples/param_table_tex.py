import pandas as pd
from pathlib import Path

from HOD_and_cosmo_prior_ranges import get_HOD_params_prior_range, get_cosmo_params_prior_range, get_fiducial_cosmo_params, get_fiducial_HOD_params

"""
Creates latex table of HOD and cosmological parameters and their prior ranges.
"""
D13_PATH        = Path("/mn/stornext/d13/euclid_nobackup/halo/AbacusSummit/emulation_files/")
EMULPATH        = Path("/mn/stornext/d5/data/vetleav/emulation_data/TPCF_HOD_and_cosmo")
DATASET_NAMES   = ["train", "test", "val"]

# HOD parameters used in the emulation
HOD_PARAM_NAMES = ["log10M1", "sigma_logM", "kappa", "alpha", "log10_ng"]
COSMO_PARAM_NAMES = ['N_eff', 'alpha_s', 'ns', 'sigma8', 'w0', 'wa', 'wb', 'wc']





def make_latex_table_full(
        ng_fixed = False,
        outpath = "/uio/hume/student-u74/vetleav/Documents/thesis/Masterthesis/masterthesis/tables",
        outfname = "param_priors.tex",
        # outpath="/uio/hume/student-u74/vetleav/Documents/thesis/ProjectedCorrelationFunctionArticle/tables",
        ):
    
    """
    Create a latex table of HOD and cosmological parameters and their prior ranges.
    Stores the table in outpath/param_priors.tex
    
    The table is formatted as follows:
    -------------------------------------
                Parameter    Prior-range  
    -------------------------------------
     HOD       Latex_name    [min,max]    
               Latex_name    [min,max]    
               ...
    -------------------------------------
    Cosmology  Latex_name    [min,max]
               Latex_name    [min,max]
               ...
    -------------------------------------
    """


    # Column labels of the table
    table_header    = [" ", "Parameter", "Fiducial value", "Prior range"]
    table_rows      = []

    # Latex labels of HOD and cosmological parameters
    HOD_param_labels = {
        "log10M1"       : r"$\log M_1$",
        "sigma_logM"    : r"$\sigma_{\log M}$",
        "kappa"         : r"$\kappa$",
        "alpha"         : r"$\alpha$",
        "log10_ng"      : r"$\log_{\bar{n}_g/(h^3\,\mathrm{Mpc}^{-3})}$"
    }
    cosmo_param_labels = {
        "N_eff"     : r"$N_\mathrm{eff}$",
        "alpha_s"   : r"$\dd n_s / \dd \ln k$",
        "ns"        : r"$n_s$",
        "sigma8"    : r"$\sigma_8$",
        "w0"        : r"$w_0$",
        "wa"        : r"$w_a$",
        "wb"        : r"$\omega_b$",
        "wc"        : r"$\omega_\mathrm{cdm}$",
    }

    # Load prior range dicts of HOD and cosmo params 
    HOD_prior_range         = get_HOD_params_prior_range()
    cosmo_prior_range       = get_cosmo_params_prior_range()
    fiducial_cosmo_params   = get_fiducial_cosmo_params()
    fiducial_HOD_params     = get_fiducial_HOD_params()


    # Fill table_rows with HOD parameters
    # for ii, name in enumerate(HOD_param_labels):
    for name in HOD_PARAM_NAMES:
        first_col            = "HOD" if name == HOD_PARAM_NAMES[0] else "" # Add "HOD" to the first row
        label                = HOD_param_labels[name]
        min_prior, max_prior = HOD_prior_range[name]
        fiducial_val         = f"{fiducial_HOD_params[name]:.3f}"
        if name == "log10_ng":
            prior_range          = f"[{min_prior:.1f}, {max_prior:.1f}]"
        else:
            prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        # min_prior            = f"{min_prior:.3f}"
        # max_prior            = f"{max_prior:.3f}"
        table_rows.append([first_col, label, fiducial_val, prior_range])

    # Fill table_rows with cosmological parameters
    for name in COSMO_PARAM_NAMES:
        first_col            = r"\hline Cosmology" if name == COSMO_PARAM_NAMES[0] else "" # Add \hline before "Cosmology" to first row
        label                = cosmo_param_labels[name]
        min_prior, max_prior = cosmo_prior_range[name]
        if fiducial_cosmo_params[name] == 0:
            fiducial_val = "0.0"
        elif fiducial_cosmo_params[name] == -1:
            fiducial_val = "-1.0"
        else:
            fiducial_val = f"{fiducial_cosmo_params[name]:.3f}"
        prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        # min_prior            = f"{min_prior:.3f}"
        # max_prior            = f"{max_prior:.3f}"
        # table_rows.append([first_col, label, min_prior, max_prior])
        table_rows.append([first_col, label, fiducial_val, prior_range])


    # Create dataframe and save to latex table.
    # Define caption and label of the table
    df = pd.DataFrame(table_rows, columns=table_header)
    caption = "HOD and cosmological parameters and their prior ranges."
    label = "tab:HOD_and_cosmo_params"
    if ng_fixed:
        label += "_ng_fixed"

    # Save to latex table
    if ng_fixed:
        outfname = outfname.replace(".tex", "_ng_fixed.tex")
    outfile = Path(f"{outpath}/{outfname}")

    # Check if outfile already exists, if so, ask if it should be overwritten
    if outfile.exists():
        print(f"{outfile} already exists.")
        input("Press Enter to overwrite or Ctrl+C to cancel.")

    # Save to latex table
    df.to_latex(
        index=False, 
        escape=False,
        buf=outfile,
        position="ht",
        column_format=" X X X rX ",
        caption=caption,
        label=label,)


def make_latex_table_HOD(
        ng_fixed = False,
        outpath = "/uio/hume/student-u74/vetleav/Documents/thesis/Masterthesis/masterthesis/tables",
        outfname = "HOD_params.tex",
        test=True,
        ):


    # Column labels of the table
    table_header    = ["Parameter", "Fiducial value", "Min. prior", "Max. prior"]
    table_rows      = []

    # Latex labels of HOD and cosmological parameters
    HOD_param_labels = {
        "log10M1"       : r"$\log M_1$",
        "sigma_logM"    : r"$\sigma_{\log M}$",
        "kappa"         : r"$\kappa$",
        "alpha"         : r"$\alpha$",
        "log10_ng"      : r"$\log{\ngavg}$"
    }
  
    # Load prior range dicts of HOD and cosmo params 
    HOD_prior_range         = get_HOD_params_prior_range()
    fiducial_HOD_params     = get_fiducial_HOD_params()


    # Fill table_rows with HOD parameters
    # for ii, name in enumerate(HOD_param_labels):
    for name in HOD_PARAM_NAMES:
        label                = HOD_param_labels[name]
        min_prior, max_prior = HOD_prior_range[name]
        if name == "log10_ng":
            min_prior          = f"{min_prior:.1f}"
            max_prior          = f"{max_prior:.1f}"
            fiducial_val         = f"{fiducial_HOD_params[name]:.2f}"

        else:
            min_prior          = f"{min_prior:.3f}"
            max_prior          = f"{max_prior:.3f}"
            fiducial_val         = f"{fiducial_HOD_params[name]:.3f}"
        table_rows.append([label, fiducial_val, min_prior, max_prior])


    # Create dataframe and save to latex table.
    # Define caption and label of the table
    df = pd.DataFrame(table_rows, columns=table_header)
    caption = "HOD parameters with their fiducial values and their prior ranges."
    label = "tab:HOD_params"
    if ng_fixed:
        label += "_ng_fixed"

    # Save to latex table
    if ng_fixed:
        outfname = outfname.replace(".tex", "_ng_fixed.tex")
    outfile = Path(f"{outpath}/{outfname}")

    # Check if outfile already exists, if so, ask if it should be overwritten
    if outfile.exists() and not test:
        print(f"{outfile} already exists.")
        input("Press Enter to overwrite or Ctrl+C to cancel.")

    # Save to latex table
    tab = df.to_latex(
        index=False, 
        escape=False,
        buf=None,
        position="ht",
        column_format=" X X X X ",
        caption=caption,
        label=label,)
    
    table_note = r"\multicolumn{4}{c}{\footnotesize Note: Number densities and masses are given in units of $h^3\,\mathrm{Mpc}^{-3}$ and $h^{-1}\,\mathrm{M_\odot}$, respectively.}"       

    # Parse the lines of the latex table to modify it 
    if test:
        lines = tab.split("\n")

        for line in lines:
            if line.startswith(r"\begin{tabular}"):
                line = line.replace(r"\begin{tabular}", r"\begin{tabularx}{\textwidth}")
            if line.startswith(r"\end{tabular}"):
                line = line.replace(r"\end{tabular}", r"\end{tabularx}")
            
    else:
        lines = tab.split("\n")
        with open(outfile, "w") as f:
            
            for line in lines:
                if line.startswith(r"\begin{tabular}"):
                    line = r"\begin{tabularx}{\textwidth}{@{}l *3{>{\centering\arraybackslash}X}@{}}"
                if line.startswith(r"\end{tabular}"):
                    line = line.replace(r"\end{tabular}", r"\end{tabularx}")
                f.write(line + "\n")
                if line.startswith(r"\bottomrule"):
                    f.write(table_note + "\n")


def make_latex_table_cosmo(
        ng_fixed = False,
        outpath = "/uio/hume/student-u74/vetleav/Documents/thesis/Masterthesis/masterthesis/tables",
        outfname = "cosmo_params.tex",
        test=True,
        ):
    
    cosmo_param_labels = {
        "N_eff"     : r"$N_\mathrm{eff}$",
        "alpha_s"   : r"$\dd n_s / \dd \ln k$",
        "ns"        : r"$n_s$",
        "sigma8"    : r"$\sigma_8$",
        "w0"        : r"$w_0$",
        "wa"        : r"$w_a$",
        "wb"        : r"$\omega_b$",
        "wc"        : r"$\omega_\mathrm{cdm}$",
    }

    # Load prior range dicts of HOD and cosmo params 
    cosmo_prior_range       = get_cosmo_params_prior_range()
    fiducial_cosmo_params   = get_fiducial_cosmo_params()


    # Column labels of the table
    table_header    = ["Parameter", "Fiducial value", "Min. prior", "Max. prior"]
    table_rows      = []


    # Fill table_rows with cosmological parameters
    for name in COSMO_PARAM_NAMES:
        label                = cosmo_param_labels[name]
        min_prior, max_prior = cosmo_prior_range[name]
        if fiducial_cosmo_params[name] == 0:
            fiducial_val = "0.0"
        elif fiducial_cosmo_params[name] == -1:
            fiducial_val = "-1.0"
        else:
            fiducial_val = f"{fiducial_cosmo_params[name]:.3f}"
        prior_range          = f"[{min_prior:.3f}, {max_prior:.3f}]"
        min_prior            = f"{min_prior:.3f}"
        max_prior            = f"{max_prior:.3f}"
        table_rows.append([label, fiducial_val, min_prior, max_prior])

    # Create dataframe and save to latex table.
    # Define caption and label of the table
    df = pd.DataFrame(table_rows, columns=table_header)
    caption = "Cosmological parameters with their fiducial values and their prior ranges."
    label = "tab:cosmo_params"
    if ng_fixed:
        label += "_ng_fixed"

    # Save to latex table
    if ng_fixed:
        outfname = outfname.replace(".tex", "_ng_fixed.tex")
    outfile = Path(f"{outpath}/{outfname}")

    # Check if outfile already exists, if so, ask if it should be overwritten
    if outfile.exists() and not test:
        print(f"{outfile} already exists.")
        input("Press Enter to overwrite or Ctrl+C to cancel.")

    # Save to latex table
    tab = df.to_latex(
        index=False, 
        escape=False,
        buf=None,
        position="ht",
        caption=caption,
        label=label,)
    
    table_note = r"\multicolumn{4}{c}{\footnotesize Note: The fiducial values correspond to the \verb|c000| simulation.}"       

    # Parse the lines of the latex table to modify it 
    if test:
        lines = tab.split("\n")

        for line in lines:
            if line.startswith(r"\begin{tabular}"):
                line = r"\begin{tabularx}{\textwidth}{@{}l *3{>{\centering\arraybackslash}X}@{}}"
            if line.startswith(r"\end{tabular}"):
                line = line.replace(r"\end{tabular}", r"\end{tabularx}")
            
    else:
        lines = tab.split("\n")
        with open(outfile, "w") as f:
            
            for line in lines:
                if line.startswith(r"\begin{tabular}"):
                    line = r"\begin{tabularx}{\textwidth}{@{}l *3{>{\centering\arraybackslash}X}@{}}"
                if line.startswith(r"\end{tabular}"):
                    line = line.replace(r"\end{tabular}", r"\end{tabularx}")
                f.write(line + "\n")
                if line.startswith(r"\bottomrule"):
                    f.write(table_note + "\n")



if __name__ == "__main__":
    # make_latex_table_full()
    make_latex_table_HOD(test=False)
    make_latex_table_cosmo(test=False)
    