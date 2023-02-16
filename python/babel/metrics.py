import numpy as np


def cks_vec_to_ks(A, x, y, z, a=0):
    # Heavily inspired by script from Sean Ressler.
    R = np.sqrt(x**2+y**2+z**2)
    r = np.sqrt(R**2 - a**2 + np.sqrt((R**2 - a**2)**2 + 4.0 * a**2 * z**2)) / np.sqrt(2.)
    A_ks = A * 0.
    SMALL = 1e-15
    sqrt_term = 2.0*r**2 - R**2 + a**2
    A_ks[0] = A[0]
    A_ks[1] = A[1] * (x*r)/sqrt_term + \
               A[2] * (y*r)/sqrt_term + \
               A[3] * z/r * (r**2+a**2)/sqrt_term
    A_ks[2] = A[1] * (x*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) + \
               A[2] * (y*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) + \
               A[3] * ((z*z)*(r**2+a**2)/(r**3 * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL)
                       - 1.0/(r*np.sqrt(1.0-z**2/r**2) + SMALL))
    A_ks[3] = A[1] * (-y/(x**2+y**2+SMALL) + a*r*x/((r**2+a**2)*sqrt_term)) + \
               A[2] * (x/(x**2+y**2+SMALL) + a*r*y/((r**2+a**2)*sqrt_term)) + \
               A[3] * (a*z/r/sqrt_term)

    return A_ks


def get_ks_pos_from_cks(x, y, z, bhspin):
    R = np.sqrt(np.power(x, 2.) + np.power(y, 2.) + np.power(z, 2.))
    r = np.sqrt(R**2 - bhspin*bhspin + np.sqrt(np.power(np.power(R, 2.) - bhspin*bhspin, 2.) +
                                               4.*bhspin*bhspin * np.power(z, 2.))) / np.sqrt(2.)
    h = np.arccos(z / r)
    p = np.arctan2(bhspin*x - r*y, bhspin*y + r*x)
    return r, h, p


def cks_inverse_metric(x, y, z, a):
    R = np.sqrt(x**2 + y**2 + z**2)
    r = np.sqrt(R**2 - a**2 + np.sqrt((R**2-a**2)**2 + 4*a**2*z**2))/np.sqrt(2.0)

    f = 2.0*r**3/(r**4 + a**2*z**2)
    l0 = -1.0
    l1 = (r*x + a*y)/(r**2 + a**2)
    l2 = (r*y - a*x)/(r**2 + a**2)
    l3 = z/r

    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]
    gi = np.zeros((4, 4, nx, ny, nz))
    gi[0][0] = -1.0 - f * l0*l0
    gi[0][1] = -f*l0*l1
    gi[0][2] = -f*l0*l2
    gi[0][3] = -f*l0*l3
    gi[1][1] = 1.0 - f*l1*l1
    gi[1][3] = -f*l1*l3
    gi[2][2] = 1.0 - f*l2*l2
    gi[2][3] = -f*l2*l3
    gi[1][2] = -f*l1*l2
    gi[3][3] = 1.0 - f*l3*l3
    gi[1][0] = gi[0][1]
    gi[2][0] = gi[0][2]
    gi[3][0] = gi[0][3]
    gi[3][1] = gi[1][3]
    gi[3][2] = gi[2][3]
    gi[2][1] = gi[1][2]

    return gi


def cks_metric(x, y, z, a):
    R = np.sqrt(x**2 + y**2 + z**2)
    r = np.sqrt(R**2 - a**2 + np.sqrt((R**2 - a**2)**2 + 4 * a**2 * z**2))/np.sqrt(2.0)

    f = 2.0 * r**3/(r**4 + a**2 * z**2)
    l0 = 1.0
    l1 = (r*x + a*y)/(r**2 + a**2)
    l2 = (r*y - a*x)/(r**2 + a**2)
    l3 = z/r

    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]

    g = np.zeros((4, 4, nx, ny, nz))
    g[0][0] = -1.0 + f * l0*l0
    g[0][1] = f*l0*l1
    g[0][2] = f*l0*l2
    g[0][3] = f*l0*l3
    g[1][1] = 1.0 + f*l1*l1
    g[1][3] = f*l1*l3
    g[2][2] = 1.0 + f*l2*l2
    g[2][3] = f*l2*l3
    g[1][2] = f*l1*l2
    g[3][3] = 1.0 + f*l3*l3
    g[1][0] = g[0][1]
    g[2][0] = g[0][2]
    g[3][0] = g[0][3]
    g[3][1] = g[1][3]
    g[3][2] = g[2][3]
    g[2][1] = g[1][2]
    return g


def get_gcov_ks(R, H, a=0):
    # calculate for ks components in ks coordinates at r, h
    gcov = np.zeros((R.shape[0], R.shape[1], 4, 4))
    cth = np.cos(H)
    sth = np.sin(H)
    s2 = sth*sth
    rho2 = R*R + a*a*cth*cth
    gcov[:, :, 0, 0] = (-1. + 2. * R / rho2)
    gcov[:, :, 0, 1] = (2. * R / rho2)
    gcov[:, :, 0, 3] = (-2. * a * R * s2 / rho2)
    gcov[:, :, 1, 0] = gcov[:, :, 0, 1]
    gcov[:, :, 1, 1] = (1. + 2. * R / rho2)
    gcov[:, :, 1, 3] = (-a * s2 * (1. + 2. * R / rho2))
    gcov[:, :, 2, 2] = rho2
    gcov[:, :, 3, 0] = gcov[:, :, 0, 3]
    gcov[:, :, 3, 1] = gcov[:, :, 1, 3]
    gcov[:, :, 3, 3] = s2 * (rho2 + a*a * s2 * (1. + 2. * R / rho2))
    return gcov


def get_gcon_eks_3d(R, H, a=0):
    n1, n2, n3 = R.shape
    gcov_ks = get_gcov_ks(R[:, :, 0], H[:, :, 0], a=a)
    gcon_ks = np.linalg.inv(gcov_ks)
    dXdx = np.zeros((n1, n2, 4, 4))  # X is eks, x is ks
    dXdx[:, :, 0, 0] = 1.
    dXdx[:, :, 1, 1] = 1. / R[:, :, 0]
    dXdx[:, :, 2, 2] = 1.
    dXdx[:, :, 3, 3] = 1.
    gcon_eks = np.einsum('abki,abkj->abij', dXdx, np.einsum('ablj,abkl->abkj', dXdx, gcon_ks))
    gcon_eks_3d = np.zeros((*R.shape, 4, 4))
    gcon_eks_3d[:, :, :, :, :] = gcon_eks[:, :, None, :, :]
    return gcon_eks_3d
