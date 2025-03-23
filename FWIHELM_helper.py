import numpy as np
from skimage.draw import ellipse
import scipy.sparse as sp
from skimage.filters import gaussian
from scipy.linalg import toeplitz


def set_geometry(xMeters, zMeters, dx, dz):

    xValues = np.arange(0, xMeters+dx, dx)
    zValues = np.arange(0, zMeters+dz, dz)
    nx = xValues.shape[0]
    nz = zValues.shape[0]

    domainShape = (xMeters, zMeters)
    spacing     = (dx, dz)
    gridSize    = (nx, nz)

    return domainShape, spacing, gridSize, xValues, zValues

def calc_nmse(xEst, xTrue):
    # nmse between two arrays, xEst is estimation of xTrue
    return np.linalg.norm(xEst-xTrue)**2 / np.linalg.norm(xTrue)**2 


def gen_filename(algPar, simulationPar, vTruePar):

    freqVector = simulationPar["freqVector"]    
    vTrueType = vTruePar["vTrueType"]
    algorithm = algPar["algorithm"]
    tvRegPar  = algPar["tvRegPar"]
    tvMethod  = algPar["tvMethod"]
    N_REC = simulationPar["N_REC"]
    N_SRC = simulationPar["N_SRC"]
    srcFreq = simulationPar["srcFrequency"]
    snr = simulationPar["SNRdB"]

    fMin = freqVector.min()
    fMax = freqVector.max()
    
    # ground truth, algorithm, tv on/off, nrec, nsrc, srcFreq, snrdb, 
    if algorithm == "ATCFWI-LESS":
        exchInterval = algPar["exchInterval"]
        fileName = algorithm+"_"+vTrueType+"_"+str(exchInterval)+"_freq="+str(fMin)+"-"+str(fMax)+"_tv="+tvMethod+"_tvRegPar="+str(tvRegPar)+"_srcFreq="+str(srcFreq)+"Hz_rec="+str(N_REC)+"_src="+str(N_SRC)+"_snr="+str(snr)
    else:
        fileName = algorithm+"_"+vTrueType+"_freq="+str(fMin)+"-"+str(fMax)+"_tv="+tvMethod+"_tvRegPar="+str(tvRegPar)+"_srcFreq="+str(srcFreq)+"Hz_rec="+str(N_REC)+"_src="+str(N_SRC)+"_snr="+str(snr)


    return fileName


def find_index_from_value(array1d, value):
    """
    Find index of an array using a value
    :param array1d: nparray array
    :param value:
    :return: index
    """
    return np.argmin(np.abs(array1d - value))


def ricker_wavelet_freq(freqDom):
    """
    Return ricker wavelet in frequency domain
    :param omega: frequency of interest 
    :param omegaP: center angular frequency (rad)
    :return: ricker wavelet in freq. domain
    """

    fs = 200  # sampling frequency of the source signal

    if 2*fs <= freqDom*2:  # ensure sampling criterion
        fs = 4*freqDom

    N_FFT = 2048
    time = np.arange(0, 1+1/fs, 1/fs)
    
    rickerTime = (1 - 2*np.pi**2*freqDom**2*(time-0.1)**2)*np.exp(-np.pi**2*freqDom**2*(time-0.1)**2)
    rickerFreq = np.fft.fft(rickerTime, N_FFT)
    freqVector = np.linspace(0, fs, N_FFT)

    return rickerFreq, freqVector


def gen_line_topology(N_REC, neighNo):
    """
    Generate adjacency matrix for line topology with no. of neighbors on left and right side of each sensor.
    1st and last sensor in line will have only neighbors on one side.

    :param N_REC: no. of receivers in array
    :param neighNo: no. of neighbors for each receiver on one side
    :return:
    """
    
    adjVec = np.zeros(N_REC)
    adjVec[1:neighNo+1] = 1  # defines neighborhood of one sensor
    adjacencyMat = toeplitz(adjVec, adjVec)
    adjacencyMat = adjacencyMat + np.eye(N_REC)

    return adjacencyMat



def check_grid_dispersion(dx, freqVector, velocityModel):    
    """
    Get maximum spacing using maximum angular frequency and minimum velocity to avoid grid dispersion.
    :param omega: maximum angular frequency (rad)
    :param velocityModel: nparray velocity model (m/s)
    :return: maximum grid spacing
    """
    freqMax = freqVector.max()  # max angular frequency of interest (rad)
    grid_points_per_wavelength = 12  # dependent of approximation order
    dxMax = np.min(velocityModel) / (freqMax * grid_points_per_wavelength)

    if dx >= dxMax:
        print("\n>> WARNING: Grid dispersion, dx should be lower than %f km." %(dxMax))
        # exit()

    return dxMax


def get_grid_points(xMeters, zMeters, dxMax):
    """
    Calculate grid, grid shape and grid spacing based on domain size and maximum grid spacing

    :param xMeters: domain width (m)
    :param zMeters: domain height (m)
    :param dxMax: maximum grid spacing (m)
    :return: Grid, grid shape, grid spacing

    """

    intervalsX = xMeters / dxMax
    intervalsX = np.ceil(intervalsX)+1
    dx = xMeters / intervalsX

    intervalsZ = zMeters / dxMax
    intervalsZ = np.ceil(intervalsZ)+1
    dz = zMeters / intervalsZ

    xValues = np.arange(0, xMeters+1, dx)
    zValues = np.arange(0, zMeters+1, dz)

    N_GRID_X = np.shape(xValues)[0]
    N_GRID_Z = np.shape(zValues)[0]

    return xValues, zValues, N_GRID_X, N_GRID_Z, dx, dz


def gen_ground_truth(simulationPar, vTruePar, **kwargs):

    """
    Generate velocity model
    :param modelType: string layer|anomaly|homogenous
    :param gridPtsX: grid shape along width
    :param gridPtsZ: grid shape along depth
    :param xValues: grid along width
    :param zValues: grid along depth
    :param vp: velocity
    :param kwargs: pass arguments based on modelType
    :return: velocityModel
    """

    modelType = vTruePar["vTrueType"]

    if modelType == "marmousi":
        # load benchmark model
        velocityModel, simulationPar = load_benchmark_model(simulationPar, benchModel=modelType)
    else:
        gridSize = simulationPar["gridSize"]
        xValues = simulationPar["xValues"]
        zValues = simulationPar["zValues"]
        vMin = vTruePar["vMin"]

        N_GRID_X, N_GRID_Z = gridSize

        velocityModel = np.zeros((N_GRID_X, N_GRID_Z))

        if modelType == "layer":
            layerDepths = kwargs["layerDepths"]
            idStart = 0
            for layerId, layer in enumerate(layerDepths):
                idz = find_index_from_value(zValues, layer)
                velocityModel[:, idStart:idz] = vMin[layerId]
                idStart = idz
            
            velocityModel[:, idStart:] = vMin[-1]  # last layer

        elif modelType == "ellipse":
            velocityModel = np.full((N_GRID_X, N_GRID_Z), min(vMin))
            vpAnomaly = vTruePar["vpAnomaly"]
            anomCenter = vTruePar["anomCenter"]
            anomRadius = vTruePar["anomRadius"]
            cidx = find_index_from_value(xValues, anomCenter[0])
            cidz = find_index_from_value(zValues, anomCenter[1])
            radiusx = find_index_from_value(xValues, anomRadius[0])
            radiusz = find_index_from_value(zValues, anomRadius[1])
            rr, cc = ellipse(cidx, cidz, radiusx, radiusz)
            velocityModel[rr, cc] = vpAnomaly

        elif modelType == "homogeneous":
            velocityModel = np.full((N_GRID_X, N_GRID_Z), min(vMin))  # TODO: substitute min(vp)

        elif modelType == "ellipse_grad":
            vMax = vTruePar["vMax"]
            vpAnomaly = vTruePar["vpAnomaly"]
            anomCenter = vTruePar["anomCenter"]
            anomRadius = vTruePar["anomRadius"]

            velocityModel = np.linspace(vMin, vMax, N_GRID_Z)
            velocityModel = np.tile(velocityModel, (N_GRID_X, 1))

            cidx = find_index_from_value(xValues, anomCenter[0])
            cidz = find_index_from_value(zValues, anomCenter[1])
            radiusx = find_index_from_value(xValues, anomRadius[0])
            radiusz = find_index_from_value(zValues, anomRadius[1])
            rr, cc = ellipse(cidx, cidz, radiusx, radiusz)
            velocityModel[rr, cc] = vpAnomaly

        elif modelType == "two_ellipses":
            vMax = vTruePar["vMax"]
            vp1, vp2 = vTruePar["vpAnomaly"]
            [center1, center2] = vTruePar["anomCenter"]
            [radius1, radius2] = vTruePar["anomRadius"]

            velocityModel = np.linspace(vMin, vMax, N_GRID_Z)
            velocityModel = np.tile(velocityModel, (N_GRID_X, 1))

            # anomaly 1
            cidx = find_index_from_value(xValues, center1[0])
            cidz = find_index_from_value(zValues, center1[1])
            radiusx = find_index_from_value(xValues, radius1[0])
            radiusz = find_index_from_value(zValues, radius1[1])

            rr, cc = ellipse(cidx, cidz, radiusx, radiusz)
            velocityModel[rr, cc] = vp1

            # anomaly 2
            cidx = find_index_from_value(xValues, center2[0])
            cidz = find_index_from_value(zValues, center2[1])
            radiusx = find_index_from_value(xValues, radius2[0])
            radiusz = find_index_from_value(zValues, radius2[1])

            rr, cc = ellipse(cidx, cidz, radiusx, radiusz)
            velocityModel[rr, cc] = vp2

    return velocityModel, simulationPar


def load_benchmark_model(simulationPar, benchModel):

    if benchModel == "marmousi":

        marmousi = np.fromfile('benchmark_models/marmousi_vp.bin', np.float32)
        marmousi = np.reshape(marmousi, (2301, 751))
        
        vTrue = marmousi[::8, ::8]  # downsample model
        vTrue = vTrue[50:200, 10:70]
        vTrue = 0.6 * vTrue / 1e3  # cut model and convert velocity into km/s

        # overwrite physical dimensions
        simulationPar["gridSize"] = np.shape(vTrue)
        simulationPar["spacing"] = (10/1e3, 10/1e3)  # in km
        simulationPar["domainShape"] = (vTrue.shape[0] * simulationPar["spacing"][0], vTrue.shape[1] * simulationPar["spacing"][1])

    else:
        return 0

    return vTrue, simulationPar


def gen_starting_model(vTrue, vTruePar, **kwargs):

    mode = vTruePar["vStartType"]        
    vStart = np.zeros_like(vTrue)
    nx, nz = vTrue.shape

    if mode == 'smooth':
        smoothVal = vTruePar["vStartSmooth"]
        # smoothVal = kwargs.get('smoothDev', 10)
        vStart = gaussian(vTrue, smoothVal, preserve_range=True)  # smooth true model

    elif mode == 'background':
        vp = kwargs.get('velocity', 1000)
        vStart = vp * np.ones((nx, nz))  # just take background velocity from true model

    elif mode == 'gradient':
        # define starting model
        vMin = vTruePar["vMin"]
        vMax = vTruePar["vMax"]
        vStart = np.linspace(vMin, vMax, nz)
        vStart = np.tile(vStart, (nx, 1))

    elif mode == 'load':
        filePath = kwargs.get('filename')
        loadedData = np.load(filePath)
        vStart = loadedData["vEst"]

    return vStart

def set_sources(simulationPar):
    """
    Sources always on surface.
    :param N_SRC:
    :param xMeters:
    :param dx:
    :param dz:
    :return:
    """

    N_SRC = simulationPar["N_SRC"]
    xMeters, _ = simulationPar["domainShape"]
    dx, dz = simulationPar["spacing"]

    # source coordinates
    srcx     = np.linspace(0, xMeters-dx, N_SRC)
    srcz     = np.zeros(N_SRC)
    srcGridx = srcx/dx
    srcGridx = srcGridx.astype(int)
    srcGridz = srcz/dz
    srcGridz = srcGridz.astype(int)

    srcCoordsGrid = [srcGridx, srcGridz]
    srcCoordsGrid = np.asarray(srcCoordsGrid).T
    

    return [srcGridx, srcGridz], [srcx, srcz]


def trans_coords_to_grid(coords, spacing):

    xGrid = coords[:, 0]/spacing[0]
    xGrid = xGrid.astype(int)
    zGrid = coords[:, 1]/spacing[1]
    zGrid = zGrid.astype(int)

    # xzGrid = [xGrid, zGrid]
    # xzGrid = np.asarray(xzGrid).T

    return xGrid, zGrid


def set_receivers(simulationPar, mode):
    """
    Set receiver coordinates. Either on 'surface' or in 'crosshole' constellation, i.e. at the bottom, opposite of
    source positions.
    :param N_REC:
    :param xMeters:
    :param zMeters:
    :param dx:
    :param dz:
    :param mode:
    :return:
    """
    
    N_REC = simulationPar["N_REC"] 
    xMeters, zMeters = simulationPar["domainShape"]
    dx, dz = simulationPar["spacing"]

    # receiver coordinates
    recx = np.linspace(0, xMeters-dx, N_REC)

    if mode == 'crosshole':
        recz = (zMeters - dz) * np.ones(N_REC)
    elif mode == 'surface':
        recz = np.zeros(N_REC)
        # recz = 5*np.ones(N_REC)

    recGridx = recx/dx
    recGridx = recGridx.astype(int)
    recGridz = recz/dz
    recGridz = recGridz.astype(int)

    recCoordsGrid = [recGridx, recGridz]
    recCoordsGrid = np.asarray(recCoordsGrid).T

    return [recGridx, recGridz], [recx, recz]


def vel_to_squared_slowness(vModel):

    return 1./(vModel**2)

