import numpy as np

import FWIHELM_solver
import FWIHELM_helper
import FWIHELM_inversion
import FWIHELM_plot

import config 


if __name__ == "__main__":

    np.random.seed(1234)  # set seed state for reproducibility

    algorithm  = "ATCFWI"  # select between "FWI" or "ATCFWI"
    subsurface = "marmousi"  # select between "marmousi" or "two_ellipses"
    simName    = algorithm + "_" + subsurface
    saveData   = False

    # load simulation parameters
    algPar, vTruePar, simulationPar = config.load_config(algorithm, subsurface)

    # generate ground truth model
    vTrue, simulationPar = FWIHELM_helper.gen_ground_truth(simulationPar, vTruePar)

    # determine upper bound for spacing to avoid grid dispersion
    dxMax = FWIHELM_helper.check_grid_dispersion(simulationPar["spacing"][0], algPar["freqVector"], vTruePar["vMin"])

    # get source signal in frequency domain
    srcSignal, srcFreqbins = FWIHELM_helper.ricker_wavelet_freq(freqDom=simulationPar["srcFrequency"])

    # set up starting model
    vStart = FWIHELM_helper.gen_starting_model(vTrue, vTruePar)
    
    # set source coordinates
    srcGridCoords, srcCoords = FWIHELM_helper.set_sources(simulationPar)
    # set receiver coordinates
    recGridCoords, recCoords = FWIHELM_helper.set_receivers(simulationPar, mode="surface")
    # Generate data sampling matrix    
    samplingMatP = FWIHELM_solver.getP(recGridCoords, simulationPar)

    # convert velocity model into squared slowness model
    mTrue  = FWIHELM_helper.vel_to_squared_slowness(vTrue)
    mStart = FWIHELM_helper.vel_to_squared_slowness(vStart)

    # generate seismic measurement data
    dObs, wavefieldObs = FWIHELM_solver.gen_observed_data(mTrue, algPar["freqVector"], srcSignal, srcFreqbins, srcGridCoords, recGridCoords, simulationPar)

    if algorithm == "FWI":

        """  ===== DO FULL WAVEFORM INVERSION ===== """        

        mEst, vEst, cost, nmse, ssim = FWIHELM_inversion.do_fwi(dObs, mTrue, mStart, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, 
                                                                samplingMatP, simulationPar, algPar, vTruePar, simName=simName)
        
        # plot results
        FWIHELM_plot.plot_results(mTrue, mStart, mEst, cost, simulationPar, cmap="viridis")
        

    elif algorithm == "ATCFWI":

        """  ===== DO ADAPT-THEN-COMBINE FULL WAVEFORM INVERSION ===== """
        
        mEst, vEst, cost, nmse, ssim = FWIHELM_inversion.do_atcfwi(dObs, mTrue, mStart, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, 
                                                                   samplingMatP, simulationPar, algPar, vTruePar, simName=simName)
        
        # save results data
    if saveData == True:
        folder   = "sim_results/"
        filename = FWIHELM_helper.gen_filename(algPar, simulationPar, vTruePar)
        np.savez(folder+filename, 
                 vTrue=vTrue, vStart=vStart, vEst=vEst, mEst=mEst, 
                 cost=cost, nmse=nmse, ssim=ssim,
                 srcCoords=srcCoords, recCoords=recCoords,
                 algPar=algPar, 
                 simulationPar=simulationPar,
                 vTruePar=vTruePar)
    