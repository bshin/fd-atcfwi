import numpy as np



def get_huber_tvnorm_grad(mEst, epsilon):

    """
    Computes gradient for smoothed l1 TV-norm in 2D using Huber loss function.
    
    """

    mGradx = np.gradient(mEst, axis=0)
    mGradGradx = np.gradient(mGradx, axis=0)    
    
    mGradz = np.gradient(mEst, axis=1)
    mGradGradz = np.gradient(mGradz, axis=1)

    mGradx = mGradx.flatten()
    mGradz = mGradz.flatten()
    mGradGradx = mGradGradx.flatten()
    mGradGradz = mGradGradz.flatten()


    mGradTV = np.zeros_like(mGradz)
    # go through each pixel...
    for i in range(mGradx.shape[0]):
        norm = np.sqrt(mGradx[i]**2+mGradz[i]**2)
        if norm < epsilon:
            mGradTV[i] = -(mGradGradx[i]+mGradGradz[i])/epsilon
        else:
            mGradTV[i] = -(mGradGradx[i]+mGradGradz[i])/np.sqrt(mGradx[i]**2 + mGradz[i]**2)

    mGradTV = np.reshape(mGradTV, np.shape(mEst))

    return mGradTV


def get_epsl1_grad(mEst, epsilonTV):

    denom = np.sqrt(np.linalg.norm(mEst.flatten())**2+epsilonTV)
    
    return mEst/denom


def get_l2_tvnorm_grad(mEst):
    """
    Computes gradient of l2-TV norm in 2D. 
    Second output is the l2-norm of this gradient.

    """
    epsilon = 1e-14
    modelEstGradx = np.gradient(mEst, axis=0)
    modelEstGradz = np.gradient(mEst, axis=1)
    modelEstGradNorm = np.sqrt(modelEstGradx**2 + modelEstGradz**2 + epsilon)
    modelEstGradx = modelEstGradx / (modelEstGradNorm)
    modelEstGradz = modelEstGradz / (modelEstGradNorm)
    modelEstGradGradx = np.gradient(modelEstGradx, axis=0)
    modelEstGradGradz = np.gradient(modelEstGradz, axis=1)
    modelEstTVgrad = modelEstGradGradx + modelEstGradGradz

    return modelEstTVgrad, modelEstGradNorm


def get_l1_tvnorm_grad(mEst):
    """ 
    Gradient of l1 TV reg. according to 
    Rudin, L., S. Osher, and E. Fatemi, 1992, Nonlinear total variation based noise removal algorithms
    
    """

    # FIXME: probably wrong implementation
    
    # mGradAbs = np.sqrt(mGradx**2 + mGradz**2)
    # epsilon  = 1e-3 * np.max(mGradAbs)

    # mDiff = mEst - mRef

    mDiffGradx = np.gradient(mEst, axis=0).flatten()
    mDiffGradz = np.gradient(mEst, axis=1).flatten()
    mDiffGrad  = np.column_stack([mDiffGradx, mDiffGradz])

    epsilon = 1e-3 * np.max(np.linalg.norm(mDiffGrad, ord=1, axis=1))

    denom = np.sqrt(np.abs(mDiffGrad)**2 + epsilon)
    mDiffGrad = np.divide(mDiffGrad, np.column_stack([denom, denom]))

    numerator = np.gradient(mDiffGrad[:, 0]) + np.gradient(mDiffGrad[:, 1])  # computing divergence

    mGradTV = numerator

    return np.reshape(mGradTV, np.shape(mEst))


def get_smoothed_tvnorm_grad(mEst):

    """
    Computes gradient for smoothed l1 TV-norm in 2D using epsL1 function.
    
    """

    mGradx = np.gradient(mEst, axis=0)
    mGradz = np.gradient(mEst, axis=1)

    mGradAbs = np.abs(mGradx) + np.abs(mGradz)
    epsilon  = 1e-3 * np.max(mGradAbs)

    magnitude = np.sqrt(mGradx**2 + mGradz**2 + epsilon)

    mGradx = mGradx/magnitude
    mGradz = mGradz/magnitude

    divx = np.gradient(mGradx, axis=0)
    divz = np.gradient(mGradz, axis=1)

    mGradTV = (divx + divz)
    
    return mGradTV



def get_huber_tvnorm_grad(mEst, epsilon):

    """
    Computes gradient for smoothed l1 TV-norm in 2D using Huber loss function.
    
    """

    mGradx = np.gradient(mEst, axis=0)
    mGradGradx = np.gradient(mGradx, axis=0)    
    
    mGradz = np.gradient(mEst, axis=1)
    mGradGradz = np.gradient(mGradz, axis=1)

    mGradx = mGradx.flatten()
    mGradz = mGradz.flatten()
    mGradGradx = mGradGradx.flatten()
    mGradGradz = mGradGradz.flatten()


    mGradTV = np.zeros_like(mGradz)
    # go through each pixel...
    for i in range(mGradx.shape[0]):
        pixelNorm = np.sqrt(mGradx[i]**2+mGradz[i]**2)
        if pixelNorm < epsilon:
            mGradTV[i] = (mGradGradx[i]+mGradGradz[i])/epsilon
        else:
            mGradTV[i] = (mGradGradx[i]+mGradGradz[i])/pixelNorm

    mGradTV = np.reshape(mGradTV, np.shape(mEst))

    return -mGradTV


def get_sobolev_grad(mEst, pOrder, epsilon):
    """
    Compute gradient of Sobolev space norm according to 
    
    Kazei, V. V., Kalita, M., & Alkhalifah, T. (2017). Salt-body inversion with minimum gradient support and Sobolev space norm regularizations. 79th EAGE Conference and Exhibition 2017. 
    https://doi.org/10.3997/2214-4609.201700600

    """
    mGradx = np.gradient(mEst, axis=0)
    mGradz = np.gradient(mEst, axis=1)

    denom = (mGradx**2 + mGradz**2 + epsilon)**(1-pOrder/2)

    mGradx = mGradx/denom
    mGradz = mGradz/denom

    divx = np.gradient(mGradx, axis=0)
    divz = np.gradient(mGradz, axis=1)

    sobolevGrad = -pOrder * (divx + divz)
    
    return sobolevGrad


def get_mgs_grad(mEst, epsilon):
    """
    Get gradient of minimum gradient support regularizer. According to 

    Kazei, V. V., Kalita, M., & Alkhalifah, T. (2017). Salt-body inversion with minimum gradient support and Sobolev space norm regularizations. 
    79th EAGE Conference and Exhibition 2017. https://doi.org/10.3997/2214-4609.201700600
    """

    mGradx = np.gradient(mEst, axis=0)
    mGradz = np.gradient(mEst, axis=1)

    denom = (mGradx**2 + mGradz**2 + epsilon)**2

    mGradx = mGradx*epsilon/denom
    mGradz = mGradz*epsilon/denom

    divx = np.gradient(mGradx, axis=0)
    divz = np.gradient(mGradz, axis=1)

    mgsGrad = -2 * (divx + divz)

    return mgsGrad


def get_tikhonov_grad(mEst):
    
    """
    Calculate gradient of Tikhonov regularization term on gradient of model estimate.
    """

    mGradx = np.gradient(mEst, axis=0)
    mGradz = np.gradient(mEst, axis=1)
    divx = np.gradient(mGradx, axis=0)
    divz = np.gradient(mGradz, axis=1)

    return -(divx + divz)  # laplacian of model estimate


def huber_gradient(mEst, epsilon=1e-3):
    grad_m = np.gradient(mEst)
    grad_norm = np.sqrt(grad_m[0]**2 + grad_m[1]**2 + 1e-8)  # Avoid division by zero
    
    # Huber function: TV for large values, Tikhonov for small
    weight = np.where(grad_norm < epsilon, grad_norm / epsilon, 1)
    
    return -np.gradient(weight * grad_m)


def get_tv_grad(mEst, tvMethod, **kwargs):

    if tvMethod == "l2":
        # l2 TV norm
        vEstTVgrad, _ = get_l2_tvnorm_grad(mEst)

    elif tvMethod == "l1":
        vEstTVgrad = get_l1_tvnorm_grad(mEst)

    elif tvMethod == "epsl1": # smoothed l1 norm        
        vEstTVgrad = get_smoothed_tvnorm_grad(mEst)

    elif tvMethod == "huber":
        # huber loss FIXME: Huber seems not to work properly
        vEstTVgrad = get_huber_tvnorm_grad(mEst, epsilon=1e-3)
        # vEstTVgrad = huber_gradient(mEst, epsilon=1e-3)

    elif tvMethod == "sobolev":
        vEstTVgrad = get_sobolev_grad(mEst, pOrder=1, epsilon=1e-6)

    elif tvMethod == "mgs":
        vEstTVgrad = get_mgs_grad()

    return vEstTVgrad