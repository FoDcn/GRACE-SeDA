# -*- coding: utf-8 -*-
"""
This script is for predicting. The results will be saved as .h5 data.

Author: Junyang Gou
2023.04.05
"""
import argparse

import numpy as np
import h5py
from tqdm import tqdm

from tensorflow.keras.models import load_model


def genSampleMat(data, idx_sample_tmp):
    TimePosition_tmp = TimePosition[idx_sample_tmp, :]
    Sample_tmp = np.zeros((len(idx_sample_tmp), PatchSize, PatchSize))

    if np.mod(PatchSize, 2) == 0: # Even size, the center is defined as the left up corner of the center 4 pixels
        for i_sample in range(0, len(idx_sample_tmp)):
            idx_time_tmp = MJD == TimePosition_tmp[i_sample, 0]
            lat_c = TimePosition_tmp[i_sample, 1]
            lon_c = TimePosition_tmp[i_sample, 2]
            
            # Double-side difference --> Case odd number of pixels with clear defined central pixel
            idx_lat_tmp = np.where(abs(lat - lat_c) <= PatchSize/2*Resolution)[0]
            idx_lon_tmp = np.where(abs(lon - lon_c) <= PatchSize/2*Resolution)[0]
            
            # We first concatenate longitude, then latitude is just nan with fixed longitutde size
            Patch_tmp = data[idx_time_tmp, idx_lat_tmp[0]:idx_lat_tmp[-1]+1, idx_lon_tmp[0]:idx_lon_tmp[-1]+1]

            if len(idx_lon_tmp) < PatchSize + 1: # Close to the longitude transition
                idx_lon_side = np.where(abs(lon+np.sign(lon_c)*360 - lon_c) <= PatchSize/2*Resolution)[0]
                Patch_side =  data[idx_time_tmp, idx_lat_tmp[0]:idx_lat_tmp[-1]+1, idx_lon_side[0]:idx_lon_side[-1]+1]
                if lon_c < 0: # Close to the west side. Concatenate the side patch to left
                    Patch_tmp = np.concatenate((Patch_side, Patch_tmp), axis=2)
                elif lon_c > 0:
                    Patch_tmp = np.concatenate((Patch_tmp, Patch_side), axis=2)
                
            if len(idx_lat_tmp) < PatchSize + 1: # Close to the poles
                num_nanrow = PatchSize + 1 - len(idx_lat_tmp)
                Patch_nan = np.zeros((1, num_nanrow, PatchSize + 1)) * np.nan
                if lat_c > 0:
                    Patch_tmp = np.concatenate((Patch_nan, Patch_tmp), axis=1)
                elif lat_c < 0:
                    Patch_tmp = np.concatenate((Patch_tmp, Patch_nan), axis=1)
            
            
            # For the normal case, we remove the left and top line --> CP is the left top corner
            Patch_tmp = Patch_tmp[:, 1:, 1:]
            
            # Fill the NaNs using the mean of the patch
            Patch_mean = np.nanmean(Patch_tmp)
            if ~np.isnan(Patch_mean):
                Patch_tmp[np.isnan(Patch_tmp)] = Patch_mean
            else:
                Patch_tmp[np.isnan(Patch_tmp)] = 0
                
            Sample_tmp[i_sample, :, :] = Patch_tmp
    
    return Sample_tmp


def genSampleMat_TemporalInvariant(data, idx_sample_tmp):
    TimePosition_tmp = TimePosition[idx_sample_tmp, :]
    Sample_tmp = np.zeros((len(idx_sample_tmp), PatchSize, PatchSize))

    if np.mod(PatchSize, 2) == 0: # Even size, the center is defined as the left up corner of the center 4 pixels
        for i_sample in range(0, len(idx_sample_tmp)):
            lat_c = TimePosition_tmp[i_sample, 1]
            lon_c = TimePosition_tmp[i_sample, 2]
            
            idx_lat_tmp = np.where(abs(lat - lat_c) <= PatchSize/2*Resolution)[0]
            idx_lon_tmp = np.where(abs(lon - lon_c) <= PatchSize/2*Resolution)[0]
            
            # We first concatenate longitude, then latitude is just nan with fixed longitutde size
            Patch_tmp = data[idx_lat_tmp[0]:idx_lat_tmp[-1]+1, idx_lon_tmp[0]:idx_lon_tmp[-1]+1]

            if len(idx_lon_tmp) < PatchSize + 1: # Close to the longitude transition
                idx_lon_side = np.where(abs(lon+np.sign(lon_c)*360 - lon_c) <= PatchSize/2*Resolution)[0]
                Patch_side =  data[idx_lat_tmp[0]:idx_lat_tmp[-1]+1, idx_lon_side[0]:idx_lon_side[-1]+1]
                if lon_c < 0: # Close to the west side. Concatenate the side patch to left
                    Patch_tmp = np.concatenate((Patch_side, Patch_tmp), axis=1)
                elif lon_c > 0:
                    Patch_tmp = np.concatenate((Patch_tmp, Patch_side), axis=1)
                
            if len(idx_lat_tmp) < PatchSize + 1: # Close to the poles
                num_nanrow = PatchSize + 1 - len(idx_lat_tmp)
                Patch_nan = np.zeros((num_nanrow, PatchSize + 1)) * np.nan
                if lat_c > 0:
                    Patch_tmp = np.concatenate((Patch_nan, Patch_tmp), axis=0)
                elif lat_c < 0:
                    Patch_tmp = np.concatenate((Patch_tmp, Patch_nan), axis=0)
            
            
            # For the normal case, we remove the left and top line --> CP is the left top corner
            Patch_tmp = Patch_tmp[1:, 1:]
            
            # Fill the NaNs using the mean of the patch
            Patch_mean = np.nanmean(Patch_tmp)
            if ~np.isnan(Patch_mean):
                Patch_tmp[np.isnan(Patch_tmp)] = Patch_mean
            else:
                Patch_tmp[np.isnan(Patch_tmp)] = 0
                
            Sample_tmp[i_sample, :, :] = Patch_tmp
    
    return Sample_tmp


# %%
parser = argparse.ArgumentParser(description='Set the name of models for prediction')

parser.add_argument('--ModelID', type=str, help='Model ID')
parser.add_argument('--ModelName', type=str, help='Model name (.h5 file)')
parser.add_argument('--Weights', type=str, help='Check points name (.hdf5 file)')

args = parser.parse_args()

model_ID = args.ModelID
model_name = args.ModelName # The first two experiemnts was 1996
checkpoint_name = args.Weights

print("Model ID: " + model_ID)
print("Model name: " + model_name)
print("Check points: " + checkpoint_name)


# %%
# TODO: Please modify the paths
root_path = "[Give the path to your saved model and weights (.h5 and .hdf5)]"
datapath = "[Give the path to your raw data files (.h5)]"
scalerpath = "[Give the path to your scalers (.csv)]"

# %% Input parameters
Region = "Global_TWSA"
LatMax = 90
LatMin = -90
LonMax = 180
LonMin = -180

NumBatch = 15
BatchSize_pred = 5000
Resolution = 0.5
PatchSize = 32


model_path = root_path + model_ID + model_name
checkpoint_path = root_path + model_ID + checkpoint_name

outpath = root_path + model_ID

# %% Load data
TWSA_JPLM = np.moveaxis(h5py.File((datapath + "TWSA_JPL_2019.h5"), 'r')['TWSA'][:, :, :], [0, 1, 2], [2, 1, 0])
TWSA_WGHM = np.moveaxis(h5py.File((datapath + "TWSA_WGHM_2019.h5"), 'r')['TWSA_WGHM_aligned'][:, :, :], [0, 1, 2], [2, 1, 0])

P = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['P'][:, :, :], [0, 1, 2], [2, 1, 0])
ET = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['ET'][:, :, :], [0, 1, 2], [2, 1, 0])
Qs = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['Qs'][:, :, :], [0, 1, 2], [2, 1, 0])
Qsb = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['Qsb'][:, :, :], [0, 1, 2], [2, 1, 0])
Qsm = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['Qsm'][:, :, :], [0, 1, 2], [2, 1, 0])

MJD = h5py.File((datapath + "TWSA_JPL_2019.h5"), 'r')['MJD'][:,].T
MJD = MJD.reshape(MJD.shape[0], )

LandMask = np.genfromtxt((datapath + "LandMask.csv"), delimiter=',')

TWSA_JPLM[:, LandMask==0] = np.NaN
TWSA_WGHM[:, LandMask==0] = np.NaN
P[:, LandMask==0] = np.NaN
ET[:, LandMask==0] = np.NaN
Qs[:, LandMask==0] = np.NaN
Qsb[:, LandMask==0] = np.NaN
Qsm[:, LandMask==0] = np.NaN

lat = np.arange(89.75, -90, -0.5)
lon = np.arange(-179.75, 180, 0.5)
lon_grid, lat_grid= np.meshgrid(lon, lat)

# %% Get the valid data
grid_in = np.logical_and(np.logical_and(lon_grid >= LonMin, lon_grid <= LonMax),\
                         np.logical_and(lat_grid >= LatMin, lat_grid <= LatMax))
data_in = TWSA_JPLM[0, grid_in]
lon_in = lon_grid[grid_in]
lat_in = lat_grid[grid_in]

# Compute number of samples, get shuffle index
num_date = len(MJD)
idx_val = ~np.isnan(data_in)
lat_val = lat_in[idx_val]
lon_val = lon_in[idx_val]
num_pixel = np.sum(idx_val)
num_sample = num_date * num_pixel
num_Batch_pred = int(np.ceil(num_sample/BatchSize_pred))

TimePosition = np.vstack((np.repeat(MJD, num_pixel),\
                               np.tile(lat_val, num_date),\
                               np.tile(lon_val, num_date))).T

idx_sample = np.arange(num_sample)

# %% Compute the scalers, normalize
feature_max_list = np.genfromtxt(scalerpath + 'Global_TWSA_Scaler_max.csv', delimiter=',')
feature_min_list = np.genfromtxt(scalerpath + 'Global_TWSA_Scaler_min.csv', delimiter=',')

feature_JPLM = (TWSA_JPLM - feature_min_list[0]) / (feature_max_list[0] - feature_min_list[0])
feature_WGHM = (TWSA_WGHM - feature_min_list[1]) / (feature_max_list[1] - feature_min_list[1])

feature_P = (P - feature_min_list[2]) / (feature_max_list[2] - feature_min_list[2])
feature_ET = (ET - feature_min_list[3]) / (feature_max_list[3] - feature_min_list[3])
feature_Qs = (Qs - feature_min_list[4]) / (feature_max_list[4] - feature_min_list[4])
feature_Qsb = (Qsb - feature_min_list[5]) / (feature_max_list[5] - feature_min_list[5])
feature_Qsm = (Qsm - feature_min_list[6]) / (feature_max_list[6] - feature_min_list[6])

feature_lat =(lat_grid - feature_min_list[7]) / (feature_max_list[7] - feature_min_list[7])
feature_lon = (lon_grid - feature_min_list[8]) / (feature_max_list[8] - feature_min_list[8])

# %% Placeholder for the TWSA predictions
TWSA_pred = np.zeros(TWSA_JPLM.shape)
TWSA_pred[:] = np.nan

# %% Load the model
model = load_model(model_path, compile=False)
model.summary()
model.load_weights(checkpoint_path)

# %% Predict in batch
i_batch = 1
MJD_batch = TimePosition[:, ]

for i_batch in tqdm(range(num_Batch_pred)):
    # if np.mod(i_batch, 20) == 0:
    #     print("Finished " + str(i_batch) + " of " + str(num_Batch_pred) + "iterations...")
    if i_batch == num_Batch_pred-1: # If it is the last TFRecord
        idx_st = i_batch*BatchSize_pred
        num_SampleInFile = num_sample - i_batch*BatchSize_pred
        idx_ed = idx_st + num_SampleInFile
    else:
        idx_st = i_batch*BatchSize_pred
        num_SampleInFile = BatchSize_pred
        idx_ed = idx_st + num_SampleInFile
    
    idx_sample_tmp = idx_sample[idx_st:idx_ed]
    
    Sample_JPLM = np.expand_dims(genSampleMat(feature_JPLM, idx_sample_tmp), axis=3)
    Sample_WGHM = np.expand_dims(genSampleMat(feature_WGHM, idx_sample_tmp), axis=3)
    
    Sample_P = np.expand_dims(genSampleMat(feature_P, idx_sample_tmp), axis=3)
    Sample_ET = np.expand_dims(genSampleMat(feature_ET, idx_sample_tmp), axis=3)
    Sample_Qs = np.expand_dims(genSampleMat(feature_Qs, idx_sample_tmp), axis=3)
    Sample_Qsb = np.expand_dims(genSampleMat(feature_Qsb, idx_sample_tmp), axis=3)
    Sample_Qsm = np.expand_dims(genSampleMat(feature_Qsm, idx_sample_tmp), axis=3)
    
    Sample_Lat = np.expand_dims(genSampleMat_TemporalInvariant(feature_lat, idx_sample_tmp), axis=3)
    Sample_Lon =np.expand_dims( genSampleMat_TemporalInvariant(feature_lon, idx_sample_tmp), axis=3)
    
    Sample_tmp = np.concatenate((Sample_JPLM, Sample_WGHM, Sample_P, Sample_ET,
                                 Sample_Qs, Sample_Qsb, Sample_Qsm, Sample_Lat, Sample_Lon),
                                axis=3)
    
    
    TWSA_pred_tmp = model.predict(Sample_tmp)
    
    # Align the central pixels to the global field
    TWSA_CP_tmp = TWSA_pred_tmp[:, 15, 15, 0]
    MJD_tmp = TimePosition[idx_st:idx_ed, 0]
    lat_tmp = TimePosition[idx_st:idx_ed, 1]
    lon_tmp = TimePosition[idx_st:idx_ed, 2]
    
    idx_MJD = np.searchsorted(MJD, MJD_tmp)
    idx_lat = np.searchsorted(lat, -lat_tmp, sorter=np.arange(359, -1, -1)) # Notice the requirement for searchsorted!
    idx_lon = np.searchsorted(lon, lon_tmp)
    
    TWSA_pred[idx_MJD, idx_lat, idx_lon] = TWSA_CP_tmp
    
# %%
handle = h5py.File(outpath + 'TWSA_DS.h5', 'w')
handle.create_dataset('TWSA_DS', data=TWSA_pred)
handle.close()