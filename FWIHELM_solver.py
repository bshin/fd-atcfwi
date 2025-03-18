import numpy as np
import FWIHELM_helper
from numba import njit
from scipy.sparse.linalg import gmres
from scipy.sparse import lil_matrix, diags, kron, eye
from joblib import Parallel, delayed


def gen_observed_data(mTrue, freqVector, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, simulationPar):

    if len(freqVector) > 1:
        freqVector = freqVector.flatten()
    
    gridSize = simulationPar["gridSize"]
    dx, dz   = simulationPar["spacing"]
    SNRdB    = simulationPar["SNRdB"]

    srcGridx = srcGridCoords[0]
    srcGridz = srcGridCoords[1]
    recGridx = recGridCoords[0]
    recGridz = recGridCoords[1]

    N_SRC = srcGridx.shape[0]
    N_REC = recGridx.shape[0]

    # pre-compute observed data
    dObs = np.zeros((freqVector.shape[0], N_SRC, N_REC), dtype=complex)
    wavefieldObs = np.zeros((freqVector.shape[0], N_SRC, gridSize[0]*gridSize[1]), dtype=complex)
    

    # generate observed data for each frequency over all sources
    for freqIdx, nFreq in enumerate(freqVector):
        forwardOp = getA(mTrue.shape[0], mTrue.shape[1], dx, dz, mTrue, nFreq)
        
        results = Parallel(n_jobs=N_SRC)(delayed(gen_dobs_per_source)(mTrue, nFreq, srcSignal, srcFreqbins, srcGridx[nSrc], srcGridz[nSrc], recGridx, recGridz, forwardOp) for nSrc in range(N_SRC))
        results = np.array(results, dtype="object")

        dObsSrc = results[:, 0]
        wavefieldObsSrc = results[:, 1]
        
        for nSrc in range(N_SRC):
            dObsClean = dObsSrc[nSrc].astype("complex")
            dObsPower = np.mean(np.abs(dObsClean)**2)
            SNR = 10**(SNRdB/10)
            noiseStd = np.sqrt( dObsPower / SNR )

            dObs[freqIdx, nSrc, :] = dObsClean + noiseStd/np.sqrt(2) * (np.random.randn(N_REC) + 1j*np.random.randn(N_REC))
            wavefieldObs[freqIdx, nSrc, :] = (wavefieldObsSrc[nSrc].astype("complex")).flatten()

    print(">> measurement data generated.")

    return dObs, wavefieldObs


def gen_dobs_per_source(mModel, nFreq, srcSignal, srcFreqbins, srcGridx, srcGridz, recGridx, recGridz, forwardOp):
    gridSize = (mModel.shape[0], mModel.shape[1])
    # set up source matrix
    srcMatrix = np.zeros(gridSize, dtype=complex)
    freqIdx = FWIHELM_helper.find_index_from_value(srcFreqbins, nFreq)
    # srcValue = 1
    srcMatrix[srcGridx, srcGridz] = srcSignal[freqIdx]
    # solve Helmholtz equation for wavefield 
    # TODO: low frequencies do not work yet, why? Problems with Helmholtz solver, wavefield is not correct
    wavefieldObs = get_wavefield(mModel, srcMatrix, forwardOp) 
    dObs = wavefieldObs[recGridx, recGridz]  # complex

    return dObs, wavefieldObs


@njit()
def alpha(omega, dx, dz, velocity) -> float:
    """
    Alpha value for iterative solvers

    :param omega: frequency of interest
    :param dx: spacing in x direction
    :param dz: spacing in z direction
    :param velocity: wave velocity
    :return: alpha value
    """
    return (np.square(omega) / np.square(velocity))-(2 / np.square(dx))-(2 / np.square(dz))


def get_wavefield(mModel, srcSignal, forwardOp) -> np.ndarray:

    srcSignalFlat = srcSignal.ravel()
    wavefieldFlat, _ = gmres(forwardOp, srcSignalFlat, atol=1e-3, maxiter=1000)  # uWavefield = A^(-1) srcSignal
    wavefield = wavefieldFlat.reshape(mModel.shape)

    return wavefield


def getA(N_X, N_Z, dx, dz, mModel, freq, fs=False):
    # FIXME: free surface does not work!
    """
    Create 2D Helmholtz matrix with 2nd order Clayton-Enquist absorbing boundary.
    Ported from Matlab code: https://github.com/vkazei/fastHelmholtz/blob/master/getA.m

    :param N_X: number of points in x direction
    :param N_Z: number of points in y direction
    :param dx: grid spacing x
    :param dz: grid spacing z
    :param vModel: velocity model
    :param freq: frequency [Hz]
    :param fs: free surface flag
    :return: A sparse matrix
    """

    # m = 1/ vModel**2
    m = mModel.flatten()
    n = [N_Z, N_X]  # NOTE: first grid points in z, then in x...reverse to main implementation
    h = [dz, dx]
    omega =  2 * np.pi * freq
    k = omega * np.sqrt(m)


    # z derivative operator
    D1 = diags(np.array([1, -1]) / h[0], [0, 1], (n[0] - 1, n[0])).tocsc()
    # x derivative operator
    D2 = diags(np.array([1, -1]) / h[1], [0, 1], (n[1] - 1, n[1])).tocsc()

    # Internal points of the domain
    spy2_values = np.ones(n[1])
    spy2_values[0] = 0
    spy2_values[-1] = 0
    spy2 = diags(spy2_values, 0).tocsc()
    L11 = -kron(spy2, D1.transpose() * D1)

    spy1_values = np.ones(n[0])
    spy1_values[0] = 0
    spy1_values[-1] = 0
    spy1 = diags(spy1_values, 0).tocsc()
    L22 = -kron(D2.transpose() * D2, spy1)

    # Laplacian inside
    L = L11 + L22  # OK!

    # Boundary
    L1 = -kron(eye(n[1]), D1.transpose() * D1)
    L2 = -kron(D2.transpose() * D2, eye(n[0]))

    # 2nd order derivative along the boundary
    L_BOUND = (L2 + L1 - L11 - L22).tolil()

    # Trick to have half Helmholtz in the second variable
    a = np.ones(n)
    a[:, [0, -1]] = 0.5
    a[[0, -1], :] = 0.5
    a = a.ravel("F")  # a is the same as in matlab code

    # Boundary points of the domain
    BND = np.nonzero(a != 1)[0] 

    # Normal derivative operator
    L_n = lil_matrix(L.shape)

    # Matrix a_corners is non-zero at the corners
    a_corners = np.zeros(n)
    a_corners[0, 0] = 1         # top left
    a_corners[0, -1] = 1        # top right
    a_corners[-1, 0] = 1        # bottom left
    a_corners[-1, -1] = 1       # bottom right
    a_corners = a_corners.flatten(order="F")

    # find linear indices of corner points
    CORNERS = np.nonzero(a_corners == 1)[0]
    # adjusting 2nd derivative in the corners (distance is sqrt(2) shorter)
    L_BOUND[CORNERS, :] *= 2

    # Creating a copy of L in LIL format to modify the elements
    L_corner = lil_matrix(L.shape)

    # Upper left corner
    L_corner[0, 0] = 1
    L_corner[0, n[0] + 1] = -1
    # distribute the second derivative
    L_BOUND[0, 0] = L_BOUND[0, 0]/2
    L_BOUND[0, n[0] + 1] = L_BOUND[0, 0]

    # Lower left corner
    L_corner[n[0] - 1, n[0] - 1] = 1
    L_corner[n[0] - 1, 2 * n[0] - 2] = -1
    # distribute the second derivative
    L_BOUND[n[0] - 1, n[0] - 1] = L_BOUND[n[0] - 1, n[0] - 1]/2
    L_BOUND[n[0] - 1, 2 * n[0] - 2] = L_BOUND[n[0] - 1, n[0] - 1]

    # Lower right corner
    L_corner[np.prod(n) - 1, np.prod(n) - 1] = 1
    L_corner[np.prod(n) - 1, np.prod(n) - n[0] - 2] = -1
    # distribute the second derivative
    L_BOUND[np.prod(n) - 1, np.prod(n) - 1] = L_BOUND[np.prod(n) - 1, np.prod(n) - 1]/2
    L_BOUND[np.prod(n) - 1, np.prod(n) - n[0] - 2] = L_BOUND[np.prod(n) - 1, np.prod(n) - 1]

    # Upper right corner
    L_corner[np.prod(n) - n[0], np.prod(n) - n[0]] = 1
    L_corner[np.prod(n) - n[0], np.prod(n) - 2 * n[0] + 1] = -1
    # distribute the second derivative
    L_BOUND[np.prod(n) - n[0], np.prod(n) - n[0]] = L_BOUND[np.prod(n) - n[0], np.prod(n) - n[0]]/2
    L_BOUND[np.prod(n) - n[0], np.prod(n) - 2 * n[0] + 1] = L_BOUND[np.prod(n) - n[0], np.prod(n) - n[0]]  # L_BOUND seems correct! compared to Matlab code


    L_corner = L_corner / (np.sqrt(2) * h[0])

    # For the boundary points L is working as normal derivative
    L_n[BND, :] = L[BND, :] 
     # Assemble Helmholtz matrix with 2nd order ABC
    A = diags(k**2) + L - L_n  

    # Clayton-Enquist 1977
    # P_tt + (1/v) P_xt - (v/2) P_zz = 0
    # We translate with the rules _t -> -i * omega;
    # k^2 P + 1i * k P_x + 1/2 P_zz = 0
    # where k = omega/v.

    # Then we use prepared derivative operators
    # _x -> L_n (normal derivative), _zz -> L_BOUND

    A = A - 1j * k[0] * (L_n * h[0] - L_corner) + 0.5 * L_BOUND

    # Free surface
    if fs:
        b = np.zeros(n)
        b[0, :] = 1
        b = b.ravel()
        FSP = np.nonzero(b)[0]
        A[FSP, :] = 0
        A[:, FSP] = 0
        A = omega**2 * diags(b * m) + A

    # Check the number of grid points per wavelength
    if freq > np.min(1e3 * 1 / np.sqrt(m)) / (5 * h[0]):
        print("Warning: Dispersion! f > min(1e3 * 1 / np.sqrt(m)) / (5 * h[0]) = {}".format(
            np.min(1e3 * 1 / np.sqrt(m)) / (5 * h[0])))

    return A


def getP(recGridCoords, simulationPar):    
    # convert sampling point into 1d index    
    nx, nz = simulationPar["gridSize"]
    gridPtx = recGridCoords[0]
    gridPtz = recGridCoords[1]

    # convert 2d index (gridPtx, gridPtz) into 1d index for matrix
    idx1d = gridPtz + gridPtx * nz
    idMatrix = np.eye(nx*nz)

    P = idMatrix[:, idx1d]

    return P

