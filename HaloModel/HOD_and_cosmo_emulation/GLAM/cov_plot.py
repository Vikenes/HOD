import numpy as np 
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 
from matplotlib import gridspec

DATAPATH     = Path(f"/mn/stornext/d5/data/vetleav/HOD_AbacusData/inference_data")

def load_cov_corr(filename):
    cov     = np.load(f'{DATAPATH}/cov_{filename}.npy')
    corr    = np.load(f'{DATAPATH}/corrcoef_{filename}.npy')

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
        fig.clf()

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

    figname = Path("figures/corr.png")
    if figname.exists():
        plt.show()
    else:
        print(f"saving {figname}")
        fig.savefig(figname, 
                    bbox_inches="tight", 
                    dpi=200)
        fig.clf()

def compare_cov_sz_and_cov_GLAM(cov_GLAM, cov_OLD):
    cov_inv_GLAM = get_cov_inv(cov_GLAM) # GLAM
    cov_inv_OLD = get_cov_inv(cov_OLD) # SZ

    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(2, 2,)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])

    ax0.set_title("Covariance GLAM")
    ax1.set_title("Covariance OLD")
    ax2.set_title("Inverse covariance GLAM")
    ax3.set_title("Inverse covariance OLD")

    # im0 = ax0.imshow(np.log(cov_GLAM), origin="lower", cmap='bwr', interpolation="none")
    im0 = ax0.imshow(cov_GLAM, origin="lower", cmap='bwr', norm=LogNorm())
    im1 = ax1.imshow(cov_OLD, origin="lower", cmap='bwr', norm=LogNorm())
    im2 = ax2.imshow(cov_inv_GLAM, origin="lower", cmap='bwr')
    im3 = ax3.imshow(cov_inv_OLD, origin="lower", cmap='bwr')
    

    fig.colorbar(im0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, fraction=0.046, pad=0.04)

    fig.tight_layout()


    figname = Path("figures/compare_old_cov.png")
    if figname.exists():
        plt.show()
    else:
        print(f"saving {figname}")
        fig.savefig(figname, 
                    bbox_inches="tight", 
                    dpi=200)
        fig.clf()


def check_cov(cov):
    cond = np.linalg.cond(cov)
    eigenvalues, _ = np.linalg.eig(cov)
    print(f"{eigenvalues=}")
    print(f"Condition number: {cond}")
    print()

cov_sz, corr_sz = load_cov_corr("wp_fiducial_sz")
cov_GLAM, corr_GLAM = load_cov_corr("wp_fiducial_MGGLAM")


# check_cov(cov_sz)
# check_cov(cov_GLAM)
# plot_cov_cov_inv(cov_GLAM)
# plot_corr(corr_GLAM)
compare_cov_sz_and_cov_GLAM(cov_GLAM, cov_sz)


exit()



cond = np.linalg.cond(cov_sz)
eigenvalues, _ = np.linalg.eig(cov_sz)
print(f"{eigenvalues=}")
print(f"Condition number: {cond}")
exit()
cov_rz, corr_rz = load_cov_corr("wp_fiducial_rz")
# print(cov_sz.shape, corr_sz.shape)
# compare_cov_sz_and_cov_rz(cov_sz, cov_rz)
exit()
fig = plt.figure(figsize=(12, 5))
gs  = gridspec.GridSpec(1, 2,)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_title("Covariance matrix")
ax1.set_title("Inverse covariance matrix")
im0 = ax0.imshow(cov, origin="lower", cmap='bwr')
im1 = ax1.imshow(cov_inv, origin="lower", cmap='bwr')
# im0 = ax0.imshow(cov_inv, origin="lower", cmap='bwr', vmin=-1e17, vmax=-1e14)
# im1 = ax1.imshow(cov_inv, origin="lower", cmap='bwr', vmin=1e14, vmax=1e17)

fig.colorbar(im0, fraction=0.046, pad=0.04)
fig.colorbar(im1, fraction=0.046, pad=0.04) #, ticks=np.linspace(-1, 1, 5))

# plt.savefig(
#     "/uio/hume/student-u74/vetleav/Documents/thesis/ParameterInference/figures/likelihood_tests/cov_and_cov_inverse.png",
#     bbox_inches="tight",
#     dpi=200,
#     pad_inches=0.05,
# )