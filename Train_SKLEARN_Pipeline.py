# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:42:04 2026

@author: GKoinig
"""

from SKLEARN_PIPELINES_runtime_threshold import Shallow_NN, SVM_RBF, SVM_Linear

paths = ["PE.mat", "PP.mat", "PET.mat"]

clf = Shallow_NN(confidence_threshold=0.70)
clf.train(paths, name="PE_PP_PET_SNN.joblib")

svm = SVM_RBF(C=10.0, confidence_threshold=0.70)
svm.train(paths, name="PE_PP_PET_SVM_RBF.joblib")

svm = SVM_Linear(C=10.0, confidence_threshold=0.70)
svm.train(paths, name="PE_PP_PET_SVM_Linear.joblib")

#%%
from SKLEARN_PIPELINES import Shallow_NN

paths = [
    "PE.mat",
    "PP.mat",
    "PET.mat",
]
paths = ["C:/Users/Technikum-Admin/Desktop/Gerald/SenSoRTC-17June26/Recordings_NIR/nir_raw_spectral_20260616_162950_362_to_20260616_162952_395_chunk00000_stopped.npy"]
clf = Shallow_NN()
clf.train(
    paths,
    name="DWRL_ADJUSTABLE_CONF_TEST.joblib",
)

#%%
from SKLEARN_PIPELINES import SVM_RBF

paths = [
    "PE.mat",
    "PP.mat",
    "PET.mat",
]
#paths = ["C:/Users/Technikum-Admin/Desktop/Gerald/SenSoRTC-17June26/Recordings_NIR/nir_raw_spectral_20260616_162950_362_to_20260616_162952_395_chunk00000_stopped.npy"]
clf = SVM_RBF()
clf.train(
    paths,
    name="SVM_LIN_DWRL_ADJUSTABLE_CONF_TEST.joblib",
)

#%%
# Fast high-LPS NIR classifier:
# mRMR reduces the spectrum from 212/220 bands to 30 selected bands.
from SKLEARN_PIPELINES_mrmr import SVM_Linear_MRMR, Shallow_NN_MRMR

paths = [
    "PE.mat",
    "PP.mat",
    "PET.mat",
]

clf = SVM_Linear_MRMR(
    n_bands_select=30,
    redundancy_weight=1.0,
    C=1.0,
    confidence_threshold=0.70,
)
clf.train(
    paths,
    name="PE_PP_PET_SVM_LINEAR_MRMR30.joblib",
)

#%% Optional: small neural net after mRMR, usually slower than LinearSVM
from SKLEARN_PIPELINES import Shallow_NN_MRMR
paths = ["C:/Users/Technikum-Admin/Desktop/Gerald/SenSoRTC-NIR/Recordings_NIR/nir_raw_spectral_20260616_162950_362_to_20260616_162952_395_chunk00000_stopped_cropped.npy"]

clf = Shallow_NN_MRMR(
    n_bands_select=30,
    redundancy_weight=1.0,
    hidden_layer_sizes=(24,),
    confidence_threshold=0.70,
)
clf.train(
    paths,
    name="DWRL_TEST_SNN_MRMR30.joblib",
)

#%% Optional: small neural net after mRMR, usually slower than LinearSVM
from SKLEARN_PIPELINES import Shallow_NN_MRMR
paths = ["C:/Users/Technikum-Admin/Desktop/Gerald/SenSoRTC-NIR/NIR/PP.xlsx"]

clf = Shallow_NN_MRMR(
    n_bands_select=30,
    redundancy_weight=1.0,
    hidden_layer_sizes=(24,),
    confidence_threshold=0.70,
)
clf.train(
    paths,
    name="DWRL_PP_TEST_SNN_MRMR30.joblib",
)