# -*- coding: utf-8 -*-
"""
Loading the raw data and do preprocessing. The processed samples will then saved
as .tfrecord format for further processing. The example raw data format is .h5.

Junyang Gou
2023.04.05
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from astropy.time import Time

import tensorflow as tf


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


# %% Input parameters
## TODO: Give the necessary data path
datapath = "[Give the path to your raw data files]"
outpath = "[Give your output path]"

NumBatch = 15
BatchSize = 512
Resolution = 0.5
PatchSize = 32

Region = "Global"
LatMax = 90
LatMin = -90
LonMax = 180
LonMin = -180

# %% Load data
# TODO: You will need to change the data name here
TWSA_JPLM = np.moveaxis(h5py.File((datapath + "TWSA_JPL_2019.h5"), 'r')['TWSA'][:, :, :], [0, 1, 2], [2, 1, 0])
TWSA_WGHM = np.moveaxis(h5py.File((datapath + "TWSA_WGHM_2019.h5"), 'r')['TWSA_WGHM_aligned'][:, :, :], [0, 1, 2], [2, 1, 0])

P = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['P'][:, :, :], [0, 1, 2], [2, 1, 0])
ET = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['ET'][:, :, :], [0, 1, 2], [2, 1, 0])
Qs = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['Qs'][:, :, :], [0, 1, 2], [2, 1, 0])
Qsb = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['Qsb'][:, :, :], [0, 1, 2], [2, 1, 0])
Qsm = np.moveaxis(h5py.File((datapath + "GLDAS-Noah_2019.h5"), 'r')['Qsm'][:, :, :], [0, 1, 2], [2, 1, 0])

MJD = h5py.File((datapath + "TWSA_JPL_2019.h5"), 'r')['MJD'][:,].T
MJD = MJD.reshape(MJD.shape[0], )

LandMask = np.genfromtxt((datapath + "Mask_Land.csv"), delimiter=',') # For the basin-wise model Amazonas

# Mask out the non-land pixels
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
num_file = np.ceil(num_sample / (NumBatch*BatchSize)).astype(np.int64)

TimePosition = np.vstack((np.repeat(MJD, num_pixel),\
                               np.tile(lat_val, num_date),\
                               np.tile(lon_val, num_date))).T

np.random.seed(1996)
idx_sample_shuffle = np.arange(num_sample)
np.random.shuffle(idx_sample_shuffle)

# %% Roboust normalization based on 4-sigma
feature_max_list = [np.nanpercentile(TWSA_JPLM[:, grid_in], 99.99),
                    np.nanpercentile(TWSA_WGHM[:, grid_in], 99.99),
                    np.nanpercentile(P[:, grid_in], 99.99),
                    np.nanpercentile(ET[:, grid_in], 99.99),
                    np.nanpercentile(Qs[:, grid_in], 99.99),
                    np.nanpercentile(Qsb[:, grid_in], 99.99),
                    np.nanpercentile(Qsm[:, grid_in], 99.99),
                    90,
                    180]

feature_min_list = [np.nanpercentile(TWSA_JPLM[:, grid_in], 0.01),
                    np.nanpercentile(TWSA_WGHM[:, grid_in], 0.01),
                    np.nanpercentile(P[:, grid_in], 0.01),
                    np.nanpercentile(ET[:, grid_in], 0.01),
                    np.nanpercentile(Qs[:, grid_in], 0.01),
                    np.nanpercentile(Qsb[:, grid_in], 0.01),
                    np.nanpercentile(Qsm[:, grid_in], 0.01),
                    -90,
                    -180]

np.savetxt(outpath + Region + "_Scaler_max.csv", np.array(feature_max_list), delimiter=",")
np.savetxt(outpath + Region + "_Scaler_min.csv", np.array(feature_min_list), delimiter=",")

feature_JPLM = (TWSA_JPLM - feature_min_list[0]) / (feature_max_list[0] - feature_min_list[0])
feature_WGHM = (TWSA_WGHM - feature_min_list[1]) / (feature_max_list[1] - feature_min_list[1])

feature_P = (P - feature_min_list[2]) / (feature_max_list[2] - feature_min_list[2])
feature_ET = (ET - feature_min_list[3]) / (feature_max_list[3] - feature_min_list[3])
feature_Qs = (Qs - feature_min_list[4]) / (feature_max_list[4] - feature_min_list[4])
feature_Qsb = (Qsb - feature_min_list[5]) / (feature_max_list[5] - feature_min_list[5])
feature_Qsm = (Qsm - feature_min_list[6]) / (feature_max_list[6] - feature_min_list[6])

feature_lat =(lat_grid - feature_min_list[7]) / (feature_max_list[7] - feature_min_list[7])
feature_lon = (lon_grid - feature_min_list[8]) / (feature_max_list[8] - feature_min_list[8])


# %%
# Compute the number of samples in one file
for i_file in tqdm(range(num_file)):
    if i_file == num_file-1: # If it is the last TFRecord
        idx_st = i_file*BatchSize*NumBatch
        num_SampleInFile = num_sample - i_file*BatchSize*NumBatch
        idx_ed = idx_st + num_SampleInFile
    else:
        idx_st = i_file*BatchSize*NumBatch
        num_SampleInFile = BatchSize*NumBatch
        idx_ed = idx_st + num_SampleInFile

    idx_sample_tmp = idx_sample_shuffle[idx_st:idx_ed]

    # For JPLM and WGHM, we need both normalized and unmormalized data
    Sample_JPLM_ori = genSampleMat(TWSA_JPLM, idx_sample_tmp)
    Sample_WGHM_ori = genSampleMat(TWSA_WGHM, idx_sample_tmp)

    Sample_JPLM = genSampleMat(feature_JPLM, idx_sample_tmp)
    Sample_WGHM = genSampleMat(feature_WGHM, idx_sample_tmp)

    # For the others, we just need normalized data
    Sample_P = genSampleMat(feature_P, idx_sample_tmp)
    Sample_ET = genSampleMat(feature_ET, idx_sample_tmp)
    Sample_Qs = genSampleMat(feature_Qs, idx_sample_tmp)
    Sample_Qsb = genSampleMat(feature_Qsb, idx_sample_tmp)
    Sample_Qsm = genSampleMat(feature_Qsm, idx_sample_tmp)

    Sample_Lat = genSampleMat_TemporalInvariant(feature_lat, idx_sample_tmp)
    Sample_Lon = genSampleMat_TemporalInvariant(feature_lon, idx_sample_tmp)

    # Set the file name
    if i_file < 9:
        fname = outpath + Region + "_000" + str(i_file+1) + ".tfrecords"
    elif i_file < 99:
        fname = outpath + Region + "_00" + str(i_file+1) + ".tfrecords"
    elif i_file < 999:
        fname = outpath + Region + "_0" + str(i_file+1) + ".tfrecords"
    else:
        fname = outpath + Region + "_" + str(i_file+1) + ".tfrecords"

    with tf.io.TFRecordWriter(fname) as file_writer:
        for i in range(num_SampleInFile):
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "GRACE_ori": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_JPLM_ori[i, :, :].reshape(-1))),
                "WGHM_ori": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_WGHM_ori[i, :, :].reshape(-1))),

                "GRACE": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_JPLM[i, :, :].reshape(-1))),
                "WGHM": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_WGHM[i, :, :].reshape(-1))),
                "P": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_P[i, :, :].reshape(-1))),
                "ET": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_ET[i, :, :].reshape(-1))),
                "Qs": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_Qs[i, :, :].reshape(-1))),
                "Qsb": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_Qsb[i, :, :].reshape(-1))),
                "Qsm": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_Qsm[i, :, :].reshape(-1))),

                "Lat": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_Lat[i, :, :].reshape(-1))),
                "Lon": tf.train.Feature(float_list=tf.train.FloatList(value=Sample_Lat[i, :, :].reshape(-1))),

                "Shape": tf.train.Feature(int64_list = tf.train.Int64List(value=Sample_JPLM[i, :, :].shape))
            })).SerializeToString()
            file_writer.write(record_bytes)
        file_writer.close()
