import numpy as np
import FWIHELM_helper


def load_config(algorithm, subsurface):
    
    if subsurface == "two_ellipses":

        N_FWI = 50
        # Define domain parameters
        xMeters = 1.4   # domain width (km)
        zMeters = 0.5   # domain depth (km)
        dx = dz = 0.01  # cell size (km)

        domainShape, spacing, gridSize, xValues, zValues = FWIHELM_helper.set_geometry(xMeters, zMeters, dx, dz)

        vTruePar = {
            "vTrueType": "two_ellipses",  # ground truth model type: ellipse, ellipse_grad, two_ellipses, marmousi
            "vpAnomaly": [1.8, 1.6], 
            "anomCenter": np.array([[.4, .3], [1.1, 0.2]]), 
            "anomRadius": np.array([[.25, .15], [0.15, 0.1]]),
            "vMin": 1,
            "vMax": 1.5,
            "vStartType": "gradient"
        }

        simulationPar = {
            "saveModels": False,
            "N_REC": 24,
            "N_SRC": 20,
            "domainShape": domainShape,
            "spacing": spacing,
            "gridSize": gridSize,
            "xValues": xValues,
            "zValues": zValues,
            "srcFrequency": 6,  # in Hz
            "SNRdB": 100  # SNR in dB on measurements
        }


        # define FWI algorithm parameters
        if algorithm == "FWI":
            algPar = {
                "algorithm": algorithm,
                "N_FWI": N_FWI,
                "stepSize": 0.01,
                "freqVector": np.array([2, 3, 4, 5, 6, 7, 8]),
                "tvMethod": "epsl1",
                "tvRegPar": 1e-8,  # ||Dm||_1  1e-9
                "tikhGradRegPar": 2e-8,  # ||grad(m)||^2_2  4e-9
                "tikhRegPar": 1e-8,  # ||m-m_prior||^2_2  1e-9
            }

        elif algorithm == "ATCFWI":
            algPar = {
                "algorithm": algorithm,
                "N_FWI": N_FWI,
                "neighNo": 3,
                "stepSize": 0.01,
                "freqVector": np.array([2, 3, 4, 5, 6, 7, 8]),
                "tvMethod": "epsl1",
                "tvRegPar": 3.5e-10,  # ||Dm||_1 
                "tikhGradRegPar": 4e-10,  # ||grad(m)||^2_2 
                "tikhRegPar": 3e-10,  # ||m-m_prior||^2_2 
            }

    elif subsurface == "marmousi":

        N_FWI = 40

        vTruePar = {
            "vTrueType": "marmousi",  # ground truth model type: ellipse, ellipse_grad, two_ellipses, marmousi
            "vMin": 0.96,
            "vMax": 2.2,
            "vStartType": "gradient",
            "vStartSmooth": 4
        }

        simulationPar = {
            "saveModels": False,
            "N_REC": 30,
            "N_SRC": 20,
            "srcFrequency": 7,  # in Hz
            "SNRdB": 100  # SNR in dB on measurements
        }

        # define FWI algorithm parameters
        if algorithm == "FWI":
            algPar = {
                "algorithm": algorithm,
                "N_FWI": N_FWI,
                "stepSize": 0.05,
                "freqVector": np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "tvMethod": "epsl1",
                "tvRegPar": 0,  # ||Dm||_1  1e-9
                "tikhGradRegPar": 0,  # ||grad(m)||^2_2  4e-9
                "tikhRegPar": 0  # ||m-m_prior||^2_2  1e-9
            }

        elif algorithm == "ATCFWI":
            algPar = {
                "algorithm": algorithm,
                "N_FWI": N_FWI,
                "neighNo": 3,
                "stepSize": 0.05,
                "freqVector": np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "tvMethod": "epsl1",
                "tvRegPar": 0,  # ||Dm||_1  1e-9
                "tikhGradRegPar": 0,  # ||grad(m)||^2_2  4e-9
                "tikhRegPar": 0  # ||m-m_prior||^2_2  1e-9
            }
    
    return algPar, vTruePar, simulationPar