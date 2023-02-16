import numpy as np


def write_header(hfp, r, h, p, gam, a):

    N1, N2, N3 = r.shape

    hfp['dump_cadence'] = 5
    hfp['t'] = 1000

    hfp.create_group('header')
    hfp.create_group('header/geom')
    hfp.create_group('header/geom/eks')

    hfp.create_dataset("/header/metric", data=np.string_("EKS"))
    hfp['header']['n1'] = N1
    hfp['header']['n2'] = N2
    hfp['header']['n3'] = N3

    dx1 = np.diff(np.log(r[:, 0, 0])).mean()
    dx2 = np.diff(h[0, :, 0]).mean()
    dx3 = np.diff(p[0, 0, :]).mean()
    startx1 = np.log(r[0, 0, 0]) - dx1/2.
    startx2 = h[0, 0, 0] - dx2/2.
    startx3 = 0.

    # this is the left edge of the grid
    hfp['header']['geom']['startx1'] = startx1
    hfp['header']['geom']['startx2'] = startx2
    hfp['header']['geom']['startx3'] = startx3

    # this is the separation between grid zone centers
    hfp['header']['geom']['dx1'] = dx1
    hfp['header']['geom']['dx2'] = dx2
    hfp['header']['geom']['dx3'] = dx3

    hfp['header']['geom']['eks']['a'] = a

    # these give the actual boundaries
    hfp['header']['geom']['eks']['r_eh'] = 1. + np.sqrt(1. - a*a)
    hfp['header']['geom']['eks']['r_in'] = np.exp(startx1)
    hfp['header']['geom']['eks']['r_out'] = np.exp(startx1 + dx1*N1)

    hfp['header']['n_prim'] = 2+3+3
    hfp['header']['n_prims_passive'] = 0
    hfp['header']['gam'] = gam
    hfp['header']['has_electrons'] = 0  # forces Theate_unit = MP/ME   ## TODO
    hfp['header']['has_radiation'] = 0

    prim_names = [b"RHO", b"UU", b"U1", b"U2", b"U3", b"B1", b"B2", b"B3"]
    hfp['header'].create_dataset("prim_names", data=np.array(prim_names, dtype='S'))


def get_key_for_level(c_level, n_level, ti, tj, tk, di, dj, dk):
    """Warning: this will fail if abs(c_level-n_level) > 1."""

    nti = ti
    ntj = tj
    ntk = tk

    if c_level == n_level:
        return ti+di, tj+dj, tk+dk

    while c_level < n_level:
        c_level += 1
        nti = 2 * (nti + di)
        ntj = 2 * (ntj + dj)
        ntk = 2 * (ntk + dk)
        if False:
            if di < 0:
                nti += 1
            if dj < 0:
                ntj += 1
            if dk < 0:
                ntk += 1
        if c_level == n_level:
            return c_level, nti, ntj, ntk
        break

    while c_level > n_level:
        c_level -= 1
        nti = (nti + di) // 2
        ntj = (ntj + dj) // 2
        ntk = (ntk + dk) // 2
        if c_level == n_level:
            return c_level, nti, ntj, ntk
        break

    raise Exception("unable to compute key")


def get_slice_source(ntot, oddity):
    if oddity == 1:
        return slice(ntot//2, ntot)
    else:
        return slice(0, ntot//2)


def get_01_source(v, ntot, oddity):
    if oddity == 1:
        if v == 0:
            return ntot//2
        return -1
    else:
        if v == 0:
            return 0
        return ntot // 2 - 1


def map_prim_to_prim(remapped, nprm, variable_names, fluid_params):
    """TODO: should read and adjust variable names; ensure that we accurately deal with B123."""
    if nprm == 0:
        return 0, remapped
    if nprm in [1, 2, 3]:
        return nprm + 1, remapped
    if nprm == 4:
        return 1, remapped
    return nprm, remapped


def get_new_meshblock_boundary(tlevel, ti, tj, tk, di, dj, dk, nprim, nmbi, nmbj, nmbk, mb_index_map, new_meshblock, uov, B):
    """Populate new meshblock boundary."""

    # first see if we can stay on the same level
    trial_key = tlevel, ti+di, tj+dj, tk+dk
    if trial_key in mb_index_map:

        nmb = mb_index_map[trial_key]

        src_i = 0 if di == 1 else (-1 if di == -1 else slice(0, nmbi))
        src_j = 0 if dj == 1 else (-1 if dj == -1 else slice(0, nmbj))
        src_k = 0 if dk == 1 else (-1 if dk == -1 else slice(0, nmbk))

        tgt_i = -1 if di == 1 else (0 if di == -1 else slice(1, nmbi+1))
        tgt_j = -1 if dj == 1 else (0 if dj == -1 else slice(1, nmbj+1))
        tgt_k = -1 if dk == 1 else (0 if dk == -1 else slice(1, nmbk+1))

        new_meshblock[:nprim, tgt_k, tgt_j, tgt_i] = uov[:, nmb, src_k, src_j, src_i]
        new_meshblock[nprim:, tgt_k, tgt_j, tgt_i] = B[:, nmb, src_k, src_j, src_i]

        return new_meshblock

    # then see if we can go up one level
    trial_key = get_key_for_level(tlevel, tlevel-1, ti, tj, tk, di, dj, dk)
    if trial_key in mb_index_map:

        nmb = mb_index_map[trial_key]

        _, newi, newj, newk = trial_key

        oddi = (ti+di) % 2
        oddj = (tj+dj) % 2
        oddk = (tk+dk) % 2

        source_i = get_01_source(0, nmbi, oddi) if di == 1 else (get_01_source(-1, nmbi, oddi) if di == -1 else get_slice_source(nmbi, oddi))
        source_j = get_01_source(0, nmbj, oddj) if dj == 1 else (get_01_source(-1, nmbj, oddj) if dj == -1 else get_slice_source(nmbj, oddj))
        source_k = get_01_source(0, nmbk, oddk) if dk == 1 else (get_01_source(-1, nmbk, oddk) if dk == -1 else get_slice_source(nmbk, oddk))

        for slc_i in range(2):
            for slc_j in range(2):
                for slc_k in range(2):
                    target_i = -1 if di == 1 else (0 if di == -1 else slice(1+slc_i, nmbi+slc_i+1, 2))
                    target_j = -1 if dj == 1 else (0 if dj == -1 else slice(1+slc_j, nmbj+slc_j+1, 2))
                    target_k = -1 if dk == 1 else (0 if dk == -1 else slice(1+slc_k, nmbk+slc_k+1, 2))

                    new_meshblock[:nprim, target_k, target_j, target_i] = uov[:, nmb, source_k, source_j, source_i]
                    new_meshblock[nprim:, target_k, target_j, target_i] = B[:, nmb, source_k, source_j, source_i]

        return new_meshblock

    # figure out how many slices we have (corner vs. edge vs. face)
    num_slices = 3
    if di in [-1, 1]:
        num_slices -= 1
    if dj in [-1, 1]:
        num_slices -= 1
    if dk in [-1, 1]:
        num_slices -= 1

    # finally see if we can go down one level
    trial_key = get_key_for_level(tlevel, tlevel+1, ti, tj, tk, di, dj, dk)
    if trial_key in mb_index_map:

        # always need this, since we need to deal with offsets
        newlevel, newi, newj, newk = trial_key

        # handle corners
        if num_slices == 0:

            source_i = 0
            source_j = 0
            source_k = 0

            # 2x as many meshblocks at this level, adjust
            if di == -1:
                newi += 1
                source_i = nmbi - 2
            if dj == -1:
                newj += 1
                source_j = nmbj - 2
            if dk == -1:
                newk += 1
                source_k = nmbk - 2

            trial_key = newlevel, newi, newj, newk
            nmb = mb_index_map[trial_key]

            target_i = -1 if di == 1 else 0
            target_j = -1 if dj == 1 else 0
            target_k = -1 if dk == 1 else 0

            for v1 in range(2):
                for v2 in range(2):
                    for v3 in range(2):

                        contribution = uov[:, nmb, source_k + v3, source_j + v2, source_i + v1]
                        new_meshblock[:nprim, target_k, target_j, target_i] += 0.125 * contribution

                        contribution = B[:, nmb, source_k + v3, source_j + v2, source_i + v1]
                        new_meshblock[nprim:, target_k, target_j, target_i] += 0.125 * contribution

            return new_meshblock

        # now handle edges
        elif num_slices == 1:

            source_i = 0
            source_j = 0
            source_k = 0

            # 2x as many meshblocks at this level, adjust
            if di == -1:
                newi += 1
                source_i = nmbi - 2
            if dj == -1:
                newj += 1
                source_j = nmbj - 2
            if dk == -1:
                newk += 1
                source_k = nmbk - 2

            for edge_pos in range(2):

                slc_i = 0
                slc_j = 0
                slc_k = 0

                # realign targets for corners
                if di in [-1, 1]:
                    target_i = -1 if di == 1 else 0
                else:
                    target_i = slice(1+edge_pos*nmbi//2, 1+(1+edge_pos)*nmbi//2)
                    slc_i = edge_pos
                    source_i = slice(1)

                if dj in [-1, 1]:
                    target_j = -1 if dj == 1 else 0
                else:
                    target_j = slice(1+edge_pos*nmbj//2, 1+(1+edge_pos)*nmbj//2)
                    slc_j = edge_pos
                    source_j = slice(1)

                if dk in [-1, 1]:
                    target_k = -1 if dk == 1 else 0
                else:
                    target_k = slice(1+edge_pos*nmbk//2, 1+(1+edge_pos)*nmbk//2)
                    slc_k = edge_pos
                    source_k = slice(1)

                trial_key = newlevel, newi+slc_i, newj+slc_j, newk+slc_k
                nmb = mb_index_map[trial_key]

                for v1 in range(2):
                    for v2 in range(2):
                        for v3 in range(2):

                            copy_source_i = source_i
                            if type(source_i) == int:
                                copy_source_i += v1
                            else:
                                source_i = slice(v1, nmbi+v1, 2)

                            copy_source_j = source_j
                            if type(source_j) == int:
                                copy_source_j += v2
                            else:
                                source_j = slice(v2, nmbj+v2, 2)

                            copy_source_k = source_k
                            if type(source_k) == int:
                                copy_source_k += v3
                            else:
                                source_k = slice(v3, nmbk+v3, 2)

                            contribution = uov[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                            new_meshblock[:nprim, target_k, target_j, target_i] += contribution / 8.

                            contribution = B[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                            new_meshblock[nprim:, target_k, target_j, target_i] += contribution / 8.

            return new_meshblock

        # now handle faces
        elif num_slices == 2:

            source_i = 0
            source_j = 0
            source_k = 0

            # 2x as many meshblocks at this level, adjust
            if di == -1:
                newi += 1
                source_i = nmbi - 2
            if dj == -1:
                newj += 1
                source_j = nmbj - 2
            if dk == -1:
                newk += 1
                source_k = nmbk - 2

            for edge_pos_1 in range(2):
                for edge_pos_2 in range(2):

                    slc_i = 0
                    slc_j = 0
                    slc_k = 0

                    if di in [-1, 1]:
                        edge_pos_j = edge_pos_1
                        edge_pos_k = edge_pos_2
                    elif dj in [-1, 1]:
                        edge_pos_i = edge_pos_1
                        edge_pos_k = edge_pos_2
                    else:
                        edge_pos_i = edge_pos_1
                        edge_pos_j = edge_pos_2

                    # realign targets for corners
                    if di in [-1, 1]:
                        target_i = -1 if di == 1 else 0
                    else:
                        target_i = slice(1+edge_pos_i*nmbi//2, 1+(1+edge_pos_i)*nmbi//2)
                        slc_i = edge_pos_i
                        source_i = slice(1)

                    if dj in [-1, 1]:
                        target_j = -1 if dj == 1 else 0
                    else:
                        target_j = slice(1+edge_pos_j*nmbj//2, 1+(1+edge_pos_j)*nmbj//2)
                        slc_j = edge_pos_j
                        source_j = slice(1)

                    if dk in [-1, 1]:
                        target_k = -1 if dk == 1 else 0
                    else:
                        target_k = slice(1+edge_pos_k*nmbk//2, 1+(1+edge_pos_k)*nmbk//2)
                        slc_k = edge_pos_k
                        source_k = slice(1)

                    trial_key = newlevel, newi+slc_i, newj+slc_j, newk+slc_k
                    nmb = mb_index_map[trial_key]

                    for v1 in range(2):
                        for v2 in range(2):
                            for v3 in range(2):

                                copy_source_i = source_i
                                if type(source_i) == int:
                                    copy_source_i += v1
                                else:
                                    copy_source_i = slice(v1, nmbi+v1, 2)

                                copy_source_j = source_j
                                if type(source_j) == int:
                                    copy_source_j += v2
                                else:
                                    copy_source_j = slice(v2, nmbj+v2, 2)

                                copy_source_k = source_k
                                if type(source_k) == int:
                                    copy_source_k += v3
                                else:
                                    copy_source_k = slice(v3, nmbk+v3, 2)

                                contribution = uov[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                                new_meshblock[:nprim, target_k, target_j, target_i] += contribution / 8.

                                contribution = B[:, nmb, copy_source_k, copy_source_j, copy_source_i]
                                new_meshblock[nprim:, target_k, target_j, target_i] += contribution / 8.

            return new_meshblock

    # we should only have trouble populating boundaries when we extend beyond the domain. if
    # instead we end up printing a message here, there's something odd afoot.
    tvi = ti + di
    tvj = tj + dj
    tvk = tk + dk
    if tvi >= 0 and tvj >= 0 and tvk >= 0:
        pass
        # TODO, deal with this later
        #print("unable to populate boundary for", ti, tj, tk, di, dj, dk)

    return new_meshblock
