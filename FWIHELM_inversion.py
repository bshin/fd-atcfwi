import numpy as np
import FWIHELM_solver
import FWIHELM_helper
from FWIHELM_regularizers import *

from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pyprind
import sys
from skimage.metrics import structural_similarity as compute_ssim


def do_fwi(dObs, mTrue, mStart, srcSignal, srcFreqbins, 
           srcGridCoords, recGridCoords, samplingMatP, 
           simulationPar, algPar, vTruePar, simName=""):
    

    mEst = mStart.copy()
    
    srcGridx = srcGridCoords[0]
    srcGridz = srcGridCoords[1]
    recGridx = recGridCoords[0]
    recGridz = recGridCoords[1]
    
    saveModels     = simulationPar["saveModels"]
    N_SRC          = simulationPar["N_SRC"]
    gridSize       = simulationPar["gridSize"]
    domainShape    = simulationPar["domainShape"]
    tvRegPar       = algPar["tvRegPar"]
    tikhRegPar     = algPar["tikhRegPar"]
    tikhGradRegPar = algPar["tikhGradRegPar"]
    stepSize       = algPar["stepSize"]
    tvMethod       = algPar["tvMethod"]
    freqVector     = algPar["freqVector"]
    N_FWI          = algPar["N_FWI"]
    
    xMeters, zMeters = domainShape

    print(">> Starting FWI...")
    print()

    # for slowness Plots
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(10, 7))

    # outer loop: over frequencies
    bar  = pyprind.ProgBar(N_FWI * freqVector.shape[0], stream=sys.stdout, monitor=True)

    cost = np.array([])
    nmse = np.array([])
    ssim = np.array([])
    mEstSequence = np.zeros((N_FWI * freqVector.shape[0], gridSize[0], gridSize[1]))

    for freqIdx, nFreq in enumerate(freqVector):

        mGrad = np.zeros(gridSize)

        # FWI iterations
        for kIter in range(N_FWI):
            
            # TODO: put iteration into functioN!
            
            # parallelize gradient computation over sources
            results = Parallel(n_jobs=N_SRC)(delayed(get_gradient)(simulationPar, srcSignal, srcFreqbins, srcGridx, srcGridz, recGridx, recGridz, mEst, nFreq, nSrc, samplingMatP, dObs[freqIdx, nSrc, :]) for nSrc in range(N_SRC))
            results = np.array(results, dtype="object")

            mGrad    = results[:, 0]
            costTemp = results[:, 1]  # cost over sources
            # wavefieldSyn = results[:, 2]  # synthesized wavefields            

            mGrad = np.sum(mGrad, axis=0)  # accumulate gradients over sources
            mGrad = mGrad.astype("float")

            # get gradients of regularizers
            mGradTV   = get_tv_grad(mEst, tvMethod)
            mGradTikh = get_tikhonov_grad(mEst)  # squared l2 norm on gradient of slowness

            # combine gradients
            mGrad = mGrad + tvRegPar*mGradTV + tikhGradRegPar * mGradTikh + tikhRegPar*(mEst - mStart)  # add TV gradient TODO: automatic selection of TV reg. parameter

            if vTruePar["vTrueType"] == "two_ellipses":
                mGrad[0, :] = mGrad[-1, :] = mGrad[:, 0] = mGrad[:, -1] = 0  # ignore boundary values
            
            
            # conjugate gradient Update
            if kIter == 0:
                mEstNext = mEst + stepSize*mGrad / np.max(np.abs(mGrad))
                cgDirectionOld = mGrad.flatten()
                gradVecOld     = mGrad.flatten()  # matrix into vector

            else:
                gradVec     = mGrad.flatten()  # matrix into vector
                cgStepSize  = np.dot(gradVec, (gradVec-gradVecOld)) / (np.linalg.norm(gradVecOld)**2)
                cgDirection = gradVec + max([cgStepSize, 0]) * cgDirectionOld

                # model update
                mEstNext = mEst + 0.98**kIter * stepSize*np.reshape(cgDirection/np.max(np.abs(cgDirection)), gridSize)

                # update gradients, all flattened matrices
                gradVecOld = gradVec
                cgDirectionOld = cgDirection

            # take only updated m values greater than 0
            mEst[mEstNext>0] = mEstNext[mEstNext>0]

            # TODO: following saving is done for animation generation
            if saveModels == True:
                mEstSequence[freqIdx*N_FWI+kIter, :, :] = mEst.copy()
            
            # cost[freqIdx*N_FWI+kIter] = np.sum(costTemp)  # accumulate costs over frequencies for k-th iteration()            
            cost = np.append(cost, np.sum(costTemp))  # for stopping criterion
            nmse = np.append(nmse, FWIHELM_helper.calc_nmse(mEst, mTrue))
            ssim = np.append(ssim, compute_ssim(mEst, mTrue, data_range=mTrue.max()-mTrue.min()))

            # # stopping criterion
            # if np.abs(np.sum(costTemp)-costOld)/np.sum(costTemp) < 0.1:
            #     break
                       
            # intermediate plots
            axs[0].imshow(mTrue.T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[0].set_title("True slowness model")
            axs[0].set_xlabel("x direction")
            axs[0].set_ylabel("z direction")

            axs[1].imshow(mStart.T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[1].set_title("Starting slowness model")
            axs[1].set_xlabel("x direction")
            axs[1].set_ylabel("z direction")

            axs[2].imshow(mEst.T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[2].set_title("Est. slowness model")
            axs[2].set_xlabel("x direction")
            axs[2].set_ylabel("z direction")
            plt.savefig("sim_plots/"+simName+"_mEst")

            plt.figure()
            plt.imshow(mGrad.T)
            plt.colorbar()
            plt.savefig("sim_plots/"+simName+"_mGrad")
            plt.close()

            plt.figure()
            plt.plot(cost)
            plt.ylabel("Cost")
            plt.savefig("sim_plots/"+simName+"_cost")
            plt.close()

            plt.figure()
            plt.plot(nmse)
            plt.ylabel("NMSE")
            plt.savefig("sim_plots/"+simName+"_nmse")
            plt.close()

            plt.figure()
            plt.plot(ssim)
            plt.ylabel("SSIM")
            plt.savefig("sim_plots/"+simName+"_ssim")
            plt.close()
        
            # progress bar update
            bar.update()

    print(bar)

    # convert slowness model into velocity model
    vEst = np.sqrt(1./mEst)

    if saveModels == True:
        np.save("mEstSequence", mEstSequence)
        return mEst, mEstSequence, vEst, cost

    return mEst, vEst, cost, nmse, ssim




def do_atcfwi(dObs, mTrue, mStart, srcSignal, srcFreqbins, srcGridCoords, recGridCoords, 
              samplingMatP, simulationPar, algPar, vTruePar, 
              simName=""):
   
    srcGridx = srcGridCoords[0]
    srcGridz = srcGridCoords[1]
    recGridx = recGridCoords[0]
    recGridz = recGridCoords[1]
    
    saveModels = simulationPar["saveModels"]
    N_SRC      = simulationPar["N_SRC"]
    N_REC      = simulationPar["N_REC"]
    gridSize   = simulationPar["gridSize"]
    domainShape = simulationPar["domainShape"]
    tvRegPar    = algPar["tvRegPar"]
    tikhRegPar  = algPar["tikhRegPar"]
    tikhGradRegPar = algPar["tikhGradRegPar"]
    freqVector     = algPar["freqVector"]
    N_FWI          = algPar["N_FWI"]

    stepSize = algPar["stepSize"]
    tvMethod = algPar["tvMethod"]

    adjacencyMat = FWIHELM_helper.gen_line_topology(N_REC, neighNo=algPar["neighNo"])

    print(">> Starting ATC-FWI...")    
    print()
    
    xMeters, zMeters = domainShape

    mEst = np.repeat(mStart[:, :, np.newaxis], N_REC, axis=2)  # initialize with same starting model for each receiver
    mEstNew = np.zeros_like(mEst)

    # for slowness Plots
    fig, axs = plt.subplots(2, 3, sharey=True, figsize=(10, 7))

    # outer loop: over frequencies
    bar  = pyprind.ProgBar(N_FWI * freqVector.shape[0], stream=sys.stdout, monitor=True)
    cost = np.array([])    
    costRec = np.zeros((N_FWI, N_REC))
    nmseRec = np.zeros((N_FWI*freqVector.shape[0], N_REC))
    ssimRec = np.zeros((N_FWI*freqVector.shape[0], N_REC))

    for freqIdx, nFreq in enumerate(freqVector):

        mGradRec       = np.zeros((gridSize[0], gridSize[1], N_REC))
        mGradRecNew    = np.zeros((gridSize[0], gridSize[1], N_REC))
        cgDirectionOld = np.zeros((gridSize[0]*gridSize[1], N_REC))
        gradVecOld     = np.zeros((gridSize[0]*gridSize[1], N_REC))

        # FWI iterations
        for kIter in range(N_FWI):
            
            # computation of local gradients
            for nRec in range(N_REC):                

                # parallelize gradient computation over sources
                results = Parallel(n_jobs=N_SRC)(delayed(get_gradient)(simulationPar, srcSignal, srcFreqbins, srcGridx, srcGridz, recGridx[nRec], recGridz[nRec], mEst[:, :, nRec], 
                                                                       nFreq, nSrc, samplingMatP[:, nRec], dObs[freqIdx, nSrc, nRec]) for nSrc in range(N_SRC))
                results = np.array(results, dtype="object")

                mGrad    = results[:, 0]
                costTemp = results[:, 1]  # receiver-specific cost over all sources
                costRec[kIter, nRec] = np.sum(costTemp)  # sum cost over sources
                # wavefieldSyn = results[:, 2]  # synthesized wavefields

                mGrad = np.sum(mGrad, axis=0)  # accumulate gradients over sources
                mGrad = mGrad.astype("float")

                # get receiver-specific regularizer
                mGradTV   = get_tv_grad(mEst[:, :, nRec], tvMethod)
                mGradTikh = get_tikhonov_grad(mEst[:, :, nRec])  # squared l2 norm on gradient of slowness

                # combine gradients for receiver-specific gradient
                mGrad = mGrad + tvRegPar*mGradTV + tikhGradRegPar*mGradTikh + tikhRegPar*(mEst[:, :, nRec] - mStart)
                
                # mGradRec gradient per receiver
                if vTruePar["vTrueType"] == "two_ellipses":
                    mGrad[0, :] = mGrad[-1, :] = mGrad[:, 0] = mGrad[:, -1] = 0  
                    # ignore boundary values for sharp anomalies, results in better imaging performance
                
                mGradRec[:, :, nRec] = mGrad

            # fusion of gradients
            for nRec in range(N_REC):
                neighborSet = np.where(adjacencyMat[nRec, :] == 1)  # find neighbors
                
                # fusion of gradients
                mGradRecNew[:, :, nRec] = np.mean(mGradRec[:, :, neighborSet[0]], axis=2)

                # local model update

                # conjugate gradient Update TODO: move into separate function for better readability
                if kIter == 0:
                    mEstNext = mEst[:, :, nRec] + stepSize*mGradRecNew[:, :, nRec] / np.max(np.abs(mGradRecNew[:, :, nRec]))
                    cgDirectionOld[:, nRec] = mGradRecNew[:, :, nRec].flatten()  
                    gradVecOld[:, nRec]     = mGradRecNew[:, :, nRec].flatten()  # matrix into vector

                else:
                    gradVec = mGradRecNew[:, :, nRec].flatten()  # matrix into vector

                    if np.linalg.norm(gradVecOld[:, nRec]) == 0:
                        cgStepSize  = np.dot(gradVec, (gradVec-gradVecOld[:, nRec]))
                    else:
                        cgStepSize  = np.dot(gradVec, (gradVec-gradVecOld[:, nRec])) / (np.linalg.norm(gradVecOld[:, nRec])**2)

                    cgDirection = gradVec + max([cgStepSize, 0]) * cgDirectionOld[:, nRec]

                    # model update
                    if np.max(np.abs(cgDirection)) == 0:
                        mEstNext = mEst[:, :, nRec] + 0.98**kIter * stepSize*np.reshape(cgDirection, gridSize)
                    else:
                        mEstNext = mEst[:, :, nRec] + 0.98**kIter * stepSize*np.reshape(cgDirection/np.max(np.abs(cgDirection)), gridSize)

                    # update gradients, all flattened matrices
                    gradVecOld[:, nRec]     = gradVec
                    cgDirectionOld[:, nRec] = cgDirection

                # take only updated m values greater than 0                
                mEst[:, :, nRec][mEstNext>0] = mEstNext[mEstNext>0]

            # fusion of subsurface models
            for nRec in range(N_REC):
                # fusion of subsurfaces images
                neighborSet = np.where(adjacencyMat[nRec, :] == 1)  # find neighbors
                mEstNew[:, :, nRec] = np.mean(mEst[:, :, neighborSet[0]], axis=2)
                nmseRec[freqIdx*N_FWI+kIter, nRec] = FWIHELM_helper.calc_nmse(mEstNew[:, :, nRec], mTrue)
                ssimRec[freqIdx*N_FWI+kIter, nRec] = compute_ssim(mEstNew[:, :, nRec], mTrue, data_range=mEstNew[:, :, nRec].max()-mEstNew[:, :, nRec].min())

            # update local models
            mEst = mEstNew
            
            cost = np.append(cost, np.sum(costRec[kIter, :]))  # sum receiver-specific costs over receivers to obtain cost equivalent to centralized cost
                       
            # intermediate plots
            axs[0, 0].imshow(mTrue.T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[0, 0].set_title("True slowness model")
            axs[0, 0].set_xlabel("x direction")
            axs[0, 0].set_ylabel("z direction")

            axs[0, 1].imshow(mStart.T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[0, 1].set_title("Starting slowness model")
            axs[0, 1].set_xlabel("x direction")
            axs[0, 1].set_ylabel("z direction")

            axs[0, 2].imshow(mEst[:, :, 0].T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[0, 2].set_title("Est. slowness model 1")
            axs[0, 2].set_xlabel("x direction")
            axs[0, 2].set_ylabel("z direction")

            axs[1, 0].imshow(mEst[:, :, 7].T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[1, 0].set_title("Est. slowness model 8")
            axs[1, 0].set_xlabel("x direction")
            axs[1, 0].set_ylabel("z direction")

            axs[1, 1].imshow(mEst[:, :, 12].T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[1, 1].set_title("Est. slowness model 13")
            axs[1, 1].set_xlabel("x direction")
            axs[1, 1].set_ylabel("z direction")

            axs[1, 2].imshow(mEst[:, :, -1].T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max())
            axs[1, 2].set_title("Est. slowness model 24")
            axs[1, 2].set_xlabel("x direction")
            axs[1, 2].set_ylabel("z direction")

            plt.savefig("sim_plots/"+simName+"_mEst")

            plt.figure()
            plt.plot(cost)
            plt.ylabel("Cost")
            plt.savefig("sim_plots/"+simName+"_cost")
            plt.close()

            plt.figure()
            plt.plot(np.mean(nmseRec, axis=1))
            plt.ylabel("NMSE")
            plt.savefig("sim_plots/"+simName+"_nmse")
            plt.close()

            plt.figure()
            plt.plot(np.mean(ssimRec, axis=1))
            plt.ylabel("SSIM")
            plt.savefig("sim_plots/"+simName+"_ssim")
            plt.close()
        
            # progress bar update
            bar.update()

    print(bar)

    # convert slowness model into velocity model
    vEst = np.sqrt(1./mEst)

    return mEst, vEst, cost, nmseRec, ssimRec


def get_gradient(simulationPar, srcSignal, srcFreqbins, srcGridx, srcGridz, recGridx, recGridz, mEst, nFreq, nSrc, P, dObs):

    gridSize = simulationPar["gridSize"]
    spacing  = simulationPar["spacing"]
    dx, dz   = spacing
    nx, nz   = gridSize

    srcMatrix = np.zeros(gridSize, dtype=complex)
    freqIdx = FWIHELM_helper.find_index_from_value(srcFreqbins, nFreq)
    srcMatrix[srcGridx[nSrc], srcGridz[nSrc]] = srcSignal[freqIdx]
    # synthesize data
    forwardOp = FWIHELM_solver.getA(nx, nz, dx, dz, mEst, nFreq)
    wavefieldSyn = FWIHELM_solver.get_wavefield(mEst, srcMatrix, forwardOp)
    dSyn = wavefieldSyn[recGridx, recGridz]
    # compute gradient
    if P.ndim > 1:
        adjointOp, _ = gmres(np.conjugate(forwardOp).T, P@(dSyn - dObs), maxiter=1000)  # propagation of data residuals through medium, see Plessix, 2006, eq: (25)
    else:
        adjointOp, _ = gmres(np.conjugate(forwardOp).T, P*(dSyn - dObs), maxiter=1000)  # computation for local receiver-specific gradient
    fwiGrad = (2*np.pi*nFreq)**2 * np.real(wavefieldSyn.flatten()*np.conjugate(adjointOp))  # See Plessix, 2006, eq. (27)
    # update squared slowness model
    cost = np.sum(np.abs(dSyn - dObs)**2)

    return np.reshape(fwiGrad, gridSize), cost, wavefieldSyn  # return gradient for one source and one frequency

