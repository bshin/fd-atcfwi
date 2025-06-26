import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_results(mTrue, mStart, mEst, cost, simulationPar, cmap="viridis"):
    
    domainShape = simulationPar["domainShape"]
    xMeters, zMeters = domainShape

    vTrue = 1/np.sqrt(mTrue)
    vStart = 1/np.sqrt(mStart)
    vEst = 1/np.sqrt(mEst)

    # Velocity Plots
    fig, axs = plt.subplots(1, 3, sharey=True)
    
    axs[0].imshow(vTrue.T, extent=[0, xMeters, zMeters, 0], cmap=cmap, vmin=vTrue.min(), vmax=vTrue.max())
    axs[0].set_title("True velocity model")
    axs[0].set_xlabel("x direction")
    axs[0].set_ylabel("z direction")

    axs[1].imshow(vStart.T, extent=[0, xMeters, zMeters, 0], cmap=cmap, vmin=vTrue.min(), vmax=vTrue.max())
    axs[1].set_title("Starting velocity model")
    axs[1].set_xlabel("x direction")
    axs[1].set_ylabel("z direction")

    axs[2].imshow(vEst.T, extent=[0, xMeters, zMeters, 0], cmap=cmap, vmin=vTrue.min(), vmax=vTrue.max())
    axs[2].set_title("Est. velocity model")
    axs[2].set_xlabel("x direction")
    axs[2].set_ylabel("z direction")

    # Slowness Plots
    fig, axs = plt.subplots(1, 3, sharey=True)
    
    axs[0].imshow(mTrue.T, extent=[0, xMeters, zMeters, 0], cmap=cmap, vmin=mTrue.min(), vmax=mTrue.max())
    axs[0].set_title("True slowness model")
    axs[0].set_xlabel("x direction")
    axs[0].set_ylabel("z direction")

    axs[1].imshow(mStart.T, extent=[0, xMeters, zMeters, 0], cmap=cmap, vmin=mTrue.min(), vmax=mTrue.max())
    axs[1].set_title("Starting slowness model")
    axs[1].set_xlabel("x direction")
    axs[1].set_ylabel("z direction")

    axs[2].imshow(mEst.T, extent=[0, xMeters, zMeters, 0], cmap=cmap, vmin=mTrue.min(), vmax=mTrue.max())
    axs[2].set_title("Est. slowness model")
    axs[2].set_xlabel("x direction")
    axs[2].set_ylabel("z direction")

    plt.figure()
    plt.plot(cost)
    plt.ylabel("Cost")

    plt.show()


def gen_video(mEstSequence, cost, mTrue, domainShape, freqVector, N_FWI, videoName):
    """
    Generate mp4 videos of imaging reconstruction.

    """

    xMeters, zMeters = domainShape

    vidCost   = []  # for video
    vidModels = []

    costFig = plt.figure()
    axCost  = costFig.add_subplot(111)
    axCost.set_xlabel("Iteration no.")
    axCost.set_ylabel("Cost")
    axCost.set_xlim([0, N_FWI*freqVector.shape[0]])
    # axCost.set_ylim([0, 1.5e-6])

    mEstFig = plt.figure(figsize=(10, 7))
    axModels = mEstFig.add_subplot(111)
    axModels.set_xlabel("x direction [km]")
    axModels.set_ylabel("z direction [km]")
    axModels.set_title("Est. model")

    costSeq = np.array([])

    for kIter in range(freqVector.shape[0]*N_FWI):
        costSeq = np.append(costSeq, cost[kIter])
        vidFrame = axCost.plot(costSeq, color='tab:blue')
        vidCost.append(vidFrame)

        estFrame = axModels.imshow(mEstSequence[kIter].T, extent=[0, xMeters, zMeters, 0], vmin=mTrue.min(), vmax=mTrue.max(), cmap="viridis")
        vidModels.append([estFrame])

    # save animation
    modelsAnimation = animation.ArtistAnimation(mEstFig, vidModels, interval=1, blit=True)
    costAnimation   = animation.ArtistAnimation(costFig, vidCost, interval=1, blit=True)
    # bar = plt.colorbar(vidFrame)
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    modelsAnimation.save(videoName+'_mEst.mp4', writer=writer)
    costAnimation.save(videoName+'_cost.mp4', writer=writer)
    print(">> Video saved")