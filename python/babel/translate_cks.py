"""

    gnw 23/02/16

    translate athenak format hdf5 file into spherical (eKS) grid

"""

from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import numpy as np
import h5py
import sys

import athenak

# input parameters
interp_method = 'linear'
bhspin = 0.9375
fluid_gamma = 4./3
target_n1 = 192
target_n2 = 96
target_n3 = 96

nstart = 0
nend = 100000

max_radius = 100.


def get_extended(arr):
    """Extend 1D array to include 1+1 ghost zones."""
    arr2 = np.zeros(arr.size + 2)
    arr2[1:-1] = arr
    dx = arr[1] - arr[0]
    arr2[0] = arr2[1] - dx
    arr2[-1] = arr2[-2] + dx
    return arr2


if __name__ == "__main__":

    fname = sys.argv[1]

    # automatic below ... you should not need to touch anything below here
    reh = 1. + np.sqrt(1. - bhspin*bhspin)

    # load file
    print(f" - loading {fname}")

    hfp = h5py.File(fname, 'r')
    x1v = np.array(hfp['x1v'])
    x2v = np.array(hfp['x2v'])
    x3v = np.array(hfp['x3v'])
    x1f = np.array(hfp['x1f'])
    x2f = np.array(hfp['x2f'])
    x3f = np.array(hfp['x3f'])
    uov = np.array(hfp['uov'])
    B = np.array(hfp['B'])
    LogicalLocations = np.array(hfp['LogicalLocations'])
    Levels = np.array(hfp['Levels'])
    variable_names = np.array(hfp.attrs['VariableNames'])
    hfp.close()

    fluid_params = dict(fluid_gamma=fluid_gamma)

    # process input file
    min_level = int(Levels.min())
    max_level = int(Levels.max())

    max_l1_i = LogicalLocations[Levels == min_level][:, 0].max()
    max_l1_j = LogicalLocations[Levels == min_level][:, 1].max()
    max_l1_k = LogicalLocations[Levels == min_level][:, 2].max()

    new_meshblocks = []

    nprim, nmb, nmbk, nmbj, nmbi = uov.shape

    print(f'got {nprim} prims. will adjust for magnetic field to => 8.')
    nprim_all = 8

    mb_index_map = {}
    for mb in range(nmb):
        tlevel = Levels[mb]
        ti, tj, tk = LogicalLocations[mb]
        key = tlevel, ti, tj, tk
        mb_index_map[key] = mb

    # construct output grid
    print(" - getting new grid")

    def get_all_vals_unique(xs):
        all_xs = []
        for x in xs:
            all_xs += list(x)
        return np.array(sorted(list(set(all_xs))))

    all_x1s = get_all_vals_unique(x1v)
    all_x2s = get_all_vals_unique(x2v)
    all_x3s = get_all_vals_unique(x3v)

    extrema = np.abs(np.array([all_x1s.min(), all_x1s.max(), all_x2s.min(),
                               all_x2s.max(), all_x3s.min(), all_x3s.max()]))
    r_min = reh * 0.95
    r_max = min(max_radius, extrema.min() * 0.98)
    r_lin = np.logspace(np.log10(r_min), np.log10(r_max), target_n1)
    h_lin = np.linspace(0, np.pi, target_n2+1)
    h_lin = (h_lin[1:] + h_lin[:-1]) / 2.
    p_lin = np.linspace(0, 2.*np.pi, target_n3+1)[:-1]
    R_ks, H_ks, P_ks = np.meshgrid(r_lin, h_lin, p_lin, indexing='ij')

    # get CKS locations for R_ks, H_ks, P_ks
    X_cks = R_ks * np.cos(P_ks) * np.sin(H_ks) - bhspin * np.sin(P_ks) * np.sin(H_ks)
    Y_cks = R_ks * np.sin(P_ks) * np.sin(H_ks) + bhspin * np.cos(P_ks) * np.sin(H_ks)
    Z_cks = R_ks * np.cos(H_ks)

    # remap
    print(f" - remapping to {target_n1}x{target_n2}x{target_n3}")
    populated = np.zeros((target_n1, target_n2, target_n3))
    prims = np.zeros((target_n1, target_n2, target_n3, 8))

    for mbi in tqdm(mb_index_map.values()):

        if mbi < nstart or mbi >= nend:
            break

        mb_x1min = x1f[mbi].min()
        mb_x1max = x1f[mbi].max()
        mb_x2min = x2f[mbi].min()
        mb_x2max = x2f[mbi].max()
        mb_x3min = x3f[mbi].min()
        mb_x3max = x3f[mbi].max()

        mb_mask = (mb_x1min < X_cks) & (X_cks <= mb_x1max)
        mb_mask &= (mb_x2min < Y_cks) & (Y_cks <= mb_x2max)
        mb_mask &= (mb_x3min < Z_cks) & (Z_cks <= mb_x3max)
        mb_mask &= (populated == 0)

        # don't process meshblocks that don't contribute to the domain
        if np.count_nonzero(mb_mask) == 0:
            continue

        # get edges for grid interpolator
        x1e = get_extended(x1v[mbi])
        x2e = get_extended(x2v[mbi])
        x3e = get_extended(x3v[mbi])

        # get meshblock key information
        tlevel = Levels[mbi]
        ti, tj, tk = LogicalLocations[mbi]
        key = tlevel, ti, tj, tk

        # fill the center of the interpolating meshblock object
        new_meshblock = np.zeros((nprim_all, nmbi+2, nmbj+2, nmbk+2))
        new_meshblock[:nprim, 1:-1, 1:-1, 1:-1] = uov[:, mbi]
        new_meshblock[nprim:, 1:-1, 1:-1, 1:-1] = B[:, mbi]

        # populate boundaries of meshblock for interpolation
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:

                    # skip center block
                    if di == 0 and dj == 0 and dk == 0:
                        continue

                    mb_info = [tlevel, ti, tj, tk, di, dj, dk, nprim, nmbi, nmbj, nmbk]
                    new_meshblock = athenak.get_new_meshblock_boundary(*mb_info, mb_index_map,
                                                                       new_meshblock, uov, B)

        # create and use the interpolation object
        for nprm in range(nprim_all):
            prm = new_meshblock[nprm, :, :, :]
            rgi = RegularGridInterpolator((x1e, x2e, x3e), prm.transpose((2, 1, 0)),
                                          method=interp_method)

            remapped = rgi((X_cks[mb_mask], Y_cks[mb_mask], Z_cks[mb_mask]))
            outidx, outval = athenak.map_prim_to_prim(remapped, nprm, variable_names, fluid_params)

            prims[mb_mask, outidx] = outval

        # ensure we don't accidentally overwrite already-populated
        # cells (precision issues?)
        populated[mb_mask] = 1

    # output
    ofname = fname.replace(".athdf", "") + "_cks.h5"

    with h5py.File(ofname, 'w') as ohfp:
        athenak.write_header(ohfp, R_ks, H_ks, P_ks, fluid_gamma, bhspin)
        ohfp['prims'] = prims
        ohfp['populated'] = populated
        ohfp['r_ks'] = R_ks
        ohfp['h_ks'] = H_ks
        ohfp['p_ks'] = P_ks
        ohfp['x_cks'] = X_cks
        ohfp['y_cks'] = Y_cks
        ohfp['z_cks'] = Z_cks
