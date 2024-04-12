import numpy as np 
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from matplotlib import gridspec

DATAPATH     = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")


def load_cov_corr(cov_file, corr_file):
    cov     = np.load(cov_file)
    corr    = np.load(corr_file)
    return cov, corr

def get_cov_inv(cov):
    return np.linalg.inv(cov)

def print_cov_inv_pos_neg(cov_inv):
    # Print min and max of positive and negative values of the inverse covariance matrix
    cov_inv_pos = np.where(cov_inv < 0, np.nan, cov_inv)
    cov_inv_neg = np.where(cov_inv > 0, np.nan, cov_inv)
    print(f"{np.nanmin(cov_inv_pos)=:6.2e} | {np.nanmax(cov_inv_pos)=:6.2e}")
    print(f"{np.nanmin(cov_inv_neg)=:6.2e} | {np.nanmax(cov_inv_neg)=:6.2e}")

    return cov_inv_pos, cov_inv_neg

def plot_cov_cov_inv(cov):
    cov_inv = get_cov_inv(cov)

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2,)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_title("Covariance matrix")
    ax1.set_title("Inverse covariance matrix")
    im0 = ax0.imshow(cov, origin="lower", cmap='bwr')
    im1 = ax1.imshow(cov_inv, origin="lower", cmap='bwr')

    fig.colorbar(im0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, fraction=0.046, pad=0.04)

    figname = Path("figures/cov_and_cov_inverse.png")
    if figname.exists():
        plt.show()
    else:
        print(f"saving {figname}")
        fig.savefig(figname, 
                    bbox_inches="tight", 
                    dpi=200)
        # fig.clf()
        plt.close(fig)

def plot_corr(corr):
    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2,)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.set_title("unscaled colorbar")
    ax1.set_title("symmetric colorbar")

    im0 = ax0.imshow(corr, origin="lower", cmap='bwr')
    im1 = ax1.imshow(corr, origin="lower", cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(im0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, fraction=0.046, pad=0.04, ticks=np.linspace(-1, 1, 5))

    fig.suptitle("Correlation matrix")

    figname = Path("figures/corr.png")
    if figname.exists():
        plt.show()
    else:
        print(f"saving {figname}")
        fig.savefig(figname, 
                    bbox_inches="tight", 
                    dpi=200)
        # fig.clf()
        plt.close(fig)

def compare_cov_sz_and_cov_GLAM(cov_NEW, cov_GLAM):
    cov_inv_GLAM = get_cov_inv(cov_GLAM) # GLAM
    cov_inv_NEW = get_cov_inv(cov_NEW) # SZ

    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(2, 2,)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])

    ax0.set_title("Covariance Abacus small")
    ax1.set_title("Covariance GLAM")
    ax2.set_title("Inverse covariance Abacus small")
    ax3.set_title("Inverse covariance GLAM")

    # im0 = ax0.imshow(np.log(cov_GLAM), origin="lower", cmap='bwr', interpolation="none")
    im0 = ax0.imshow(cov_NEW, origin="lower", norm=LogNorm())
    im1 = ax1.imshow(cov_GLAM, origin="lower", norm=LogNorm())
    im2 = ax2.imshow(cov_inv_NEW, origin="lower")
    im3 = ax3.imshow(cov_inv_GLAM, origin="lower",)
    

    fig.colorbar(im0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, fraction=0.046, pad=0.04)

    fig.tight_layout()


    figname = Path("figures/compare_NEW_cov.png")
    if figname.exists():
        plt.show()
    else:
        print(f"saving {figname}")
        fig.savefig(figname, 
                    bbox_inches="tight", 
                    dpi=200)
        plt.close(fig)


def check_cov(cov):
    cond = np.linalg.cond(cov)
    eigenvalues, _ = np.linalg.eig(cov)
    print(f"{eigenvalues=}")
    print(f"Condition number: {cond}")
    print()

# cov_GLAM, corr_GLAM     = np.load(f'{DATAPATH}/MGGLAM/cov_wp_fiducial.npy')
cov_GLAM, corr_GLAM = load_cov_corr(
    f'{DATAPATH}/MGGLAM/cov_wp_fiducial.npy', 
    f'{DATAPATH}/MGGLAM/corrcoef_wp_fiducial.npy'
    )
cov_sz, corr_sz = load_cov_corr(
    f'{DATAPATH}/cov_wp_small.npy', 
    f'{DATAPATH}/corrcoef_wp_small.npy'
    )
cov_sz   /= 64.0 
cov_GLAM /= 8.0

# check_cov(cov_sz)
# check_cov(cov_GLAM)
# plot_cov_cov_inv(cov_sz)
# plot_corr(corr_sz)
compare_cov_sz_and_cov_GLAM(cov_sz, cov_GLAM)
# print(cov_GLAM[0])


