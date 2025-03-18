import numpy as np
import matplotlib.pyplot as plt

import FWIHELM_solver
import FWIHELM_helper
import FWIHELM_inversion
import FWIHELM_plot

if __name__ == "__main__":

    np.random.seed(1234)  # set seed state for reproducibility

    cmap     = "viridis"
    saveData = True

    algorithm = "FWI"  # FWI or ATCFWI
    N_REC     = 30  # good image: 100
    N_SRC     = 20  # good image: 30
    N_FWI     = 40  # good image: 50

    srcFrequency = 7  # in Hz
    SNR          = 100  # in dB

    # --------
    # Define domain parameters
    # --------

    xMeters = 1.4   # domain width (km)
    zMeters = 0.5   # domain depth (km)
    dx = dz = 0.01  # cell size (km)
    xValues = np.arange(0, xMeters+dx, dx)
    zValues = np.arange(0, zMeters+dz, dz)
    nx = xValues.shape[0]
    nz = zValues.shape[0]

    domainShape = (xMeters, zMeters)
    spacing     = (dx, dz)
    gridSize    = (nx, nz)

    # define frequencies
    freqVector = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])  # in Hz

    algPar = {
        "algorithm": algorithm,
        "N_FWI": N_FWI,
        "neighNo": 3,
        "exchInterval": 0,
        "stepSize": 0.05,
        # "tvRegPar": 3e-9,
        # "tikhRegPar": 1e-9
        "tvMethod": "epsl1",
        "tvRegPar": 1e-10,  # ||Dm||_1  1e-9
        "tikhGradRegPar": 0e-10,  # ||grad(m)||^2_2  4e-9
        "tikhRegPar": 0  # ||m-m_prior||^2_2  1e-9
    }

    simulationPar = {
        "saveModels": False,
        "N_REC": N_REC,
        "N_SRC": N_SRC,
        "domainShape": domainShape,
        "spacing": spacing,
        "gridSize": gridSize,
        "freqVector": freqVector,
        "xValues": xValues,
        "zValues": zValues,
        "srcFrequency": srcFrequency,  # in Hz
        "SNRdB": SNR  # SNR in dB on measurements
    }

    vTruePar = {
        "vTrueType": "marmousi",  # ground truth model type: ellipse, ellipse_grad, two_ellipses, marmousi
        "vMin": 0.96,
        "vMax": 2.2,
        "vStartType": "gradient",
        "vStartSmooth": 4
    }
    
    # get source signal in frequency domain
    srcSignal, srcFreqbins = FWIHELM_helper.ricker_wavelet_freq(freqDom=simulationPar["srcFrequency"])
    # set up groudn truth velocity model
    vTrue, simulationPar = FWIHELM_helper.gen_ground_truth(simulationPar, vTruePar)
    # check for grid dispersion
    dx, dz = simulationPar["spacing"]
    dxMax = FWIHELM_helper.check_grid_dispersion(dx, freqVector, vTrue.min())  # determine upper bound for spacing to avoid grid dispersion
    # set up starting model
    vStart = FWIHELM_helper.gen_starting_model(vTrue, vTruePar)

    # set source coordinates
    srcGridCoords, srcCoords = FWIHELM_helper.set_sources(simulationPar)
    # set receiver coordinates
    recGridCoords, recCoords = FWIHELM_helper.set_receivers(simulationPar, mode="surface")
    # Generate data sampling matrix    
    samplingMatP  = FWIHELM_solver.getP(recGridCoords, simulationPar)

    # convert velocity model into slowness model
    mTrue  = FWIHELM_helper.vel_to_squared_slowness(vTrue)
    mStart = FWIHELM_helper.vel_to_squared_slowness(vStart)

    # generate seismic measurement data
    dObs, wavefieldObs = FWIHELM_solver.gen_observed_data(mTrue, freqVector, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, simulationPar)

    if algorithm == "FWI":
        """  ===== DO FULL WAVEFORM INVERSION ===== """

        mEst, vEst, cost, nmse, ssim = FWIHELM_inversion.do_fwi(dObs, mTrue, mStart, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, 
                                                                samplingMatP, simulationPar, algPar, vTruePar,
                                                                freqVector, 
                                                                N_FWI=N_FWI,
                                                                simName="FWI_marmousi",
                                                                adaptReg=False)

        # plot results
        FWIHELM_plot.plot_results(mTrue, mStart, mEst, cost, simulationPar, cmap="viridis")

    elif algorithm == "ATCFWI":

        """  ===== DO ADAPT-THEN-COMBINE FULL WAVEFORM INVERSION ===== """
        
        mEst, vEst, cost, nmse, ssim = FWIHELM_inversion.do_atcfwi(dObs, mTrue, mStart, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, 
                                                                    samplingMatP, simulationPar, algPar, vTruePar,
                                                                    freqVector,
                                                                    N_FWI=N_FWI,
                                                                    simName="ATC_marmousi")
        
        # TODO: add plotting function for ATCFWI

    # save results data
    if saveData == True:
        folder   = "sim_results/"
        filename = FWIHELM_helper.gen_filename(algPar, simulationPar, vTruePar)
        np.savez(folder+filename, vTrue=vTrue, vStart=vStart, vEst=vEst, mEst=mEst, 
                 cost=cost, nmse=nmse, ssim=ssim,
                 srcCoords=srcCoords, recCoords=recCoords,
                 algPar=algPar, 
                 simulationPar=simulationPar,
                 vTruePar=vTruePar)

    