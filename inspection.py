#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 15:39:28 2020

@author: adnene33
"""


import os
import sys

import tensorflow as tf
from tensorflow import keras

from modules import utils as utils

# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.stats

# import requests
import matplotlib.pyplot as plt

# plt.style.use("seaborn-darkgrid")
import csv
import pandas as pd
import json

import argparse
from pathlib import Path


# import computer vision functions
import cv2
from skimage.util import img_as_ubyte
from modules.cv import scale_pixel_values as scale_pixel_values
from modules.cv import filter_gauss_images as filter_gauss_images
from modules.cv import filter_median_images as filter_median_images
from modules.cv import threshold_images as threshold_images
from modules.cv import label_images as label_images

# =============================== LOAD ARRAYS ===================================

# VALIDATION
imgs_val_input = np.load("results/werkstueck/data_a30_nikon_weiss_edit/mvtec2/SSIM/29-03-2020_10-26-12/validation/imgs_val_input.npy", allow_pickle=True)
imgs_val_pred = np.load("results/werkstueck/data_a30_nikon_weiss_edit/mvtec2/SSIM/29-03-2020_10-26-12/validation/imgs_val_pred.npy", allow_pickle=True)
resmaps_val = np.load("results/werkstueck/data_a30_nikon_weiss_edit/mvtec2/SSIM/29-03-2020_10-26-12/validation/resmaps_val.npy", allow_pickle=True)

# TEST (comment if unavailable)
imgs_test_input = np.load("results/werkstueck/data_a30_nikon_weiss_edit/mvtec2/SSIM/29-03-2020_10-26-12/test/imgs_test_input.npy", allow_pickle=True)
imgs_test_pred = np.load("results/werkstueck/data_a30_nikon_weiss_edit/mvtec2/SSIM/29-03-2020_10-26-12/test/imgs_test_pred.npy", allow_pickle=True)
resmaps_test = np.load("results/werkstueck/data_a30_nikon_weiss_edit/mvtec2/SSIM/29-03-2020_10-26-12/test/resmaps_test.npy", allow_pickle=True)

# ====================== VISUALIZE IMAGE HISTOGRAMS ===================================

def plot_img(img, title=None):
    plt.style.use("default")
    ndims = len(img.shape)
    fig = plt.figure()
    if ndims == 2:
        plt.imshow(img, cmap=plt.cm.gray)
    elif ndims == 3:
        _, _, channels = img.shape
        if channels == 3:
            plt.imshow(img)
        else:
            plt.imshow(img[:, :, 0], cmap=plt.cm.gray)
    plt.title(title)
    plt.show()
    # return fig


def hist_image(img, bins=400, range_x=None, title=None):
    # plot histogram
    plt.style.use("seaborn-darkgrid")
    img_1d = img.flatten()
    plt.figure()
    plt.hist(img_1d, bins=bins, range=range_x, density=True, stacked=True, label="image histogram")
    
    # print descriptive values
    mu = img_1d.mean()
    sigma = img_1d.std()
    minimum = np.amin(img_1d)
    maximum = np.amax(img_1d)
    print("{} \n mu = {} \n sigma = {} \n min = {}\n max = {}".format(title, mu, sigma, minimum, maximum))
    
    # compute and plot probability distribution function    
    X = np.linspace(start=minimum, stop=maximum, num=200, endpoint=True)
    pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
    plt.plot(X, pdf_x, label="pixel distribution")
    plt.title(title)
    if range_x is not None:
        plt.xlim(range_x[0], range_x[1])
    plt.legend()
    plt.show()
    
def hist_image_uint8(img, range_x=None, title=None):
    # plot histogram    
    plt.style.use("seaborn-darkgrid")
    plt.figure()
    # compute bin edges
    img_1d = img.flatten()
    d = np.diff(np.unique(img_1d)).min()
    left_of_first_bin = img_1d.min() - float(d)/2
    right_of_last_bin = img_1d.max() + float(d)/2  
    bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
    plt.hist(img_1d, bins=bins, density=True, stacked=True, label="pixel value histogram")
    
    # print descriptive values
    mu = img_1d.mean()
    sigma = img_1d.std()
    minimum = np.amin(img_1d)
    maximum = np.amax(img_1d)
    print("{} \n mu = {} \n sigma = {} \n min = {}\n max = {}".format(title, mu, sigma, minimum, maximum))
    
    # compute and plot probability distribution function    
    X = np.linspace(start=minimum, stop=maximum, num=200, endpoint=True)
    pdf_x = [scipy.stats.norm(mu, sigma).pdf(x) for x in X]
    plt.plot(X, pdf_x, label="pixel distribution")
    plt.title(title)
    if range_x is not None:
        plt.xlim(range_x[0], range_x[1])
    plt.legend()
    plt.show()
    



# plot image, pred, resmap  and pixel histogram -------------------------------

# values are float
index_val = 0
plot_img(imgs_val_input[index_val])
plot_img(imgs_val_pred[index_val])
plot_img(resmaps_val[index_val], title="resmaps_val")
hist_image(resmaps_val, title="resmaps_val")


index_test = 12
plot_img(imgs_test_input[index_test])
plot_img(imgs_test_pred[index_test])
plot_img(resmaps_test[index_test], title="resmaps_test")
hist_image(resmaps_test, title="resmaps_test")


# transform to uint8 ----------------------------------------------------------
resmaps_val_uint8 = img_as_ubyte(resmaps_val)
hist_image_uint8(resmaps_val_uint8, title="resmaps_val_uint8")

resmaps_test_uint8 = img_as_ubyte(resmaps_test)
hist_image_uint8(resmaps_test_uint8, title="resmaps_test_uint8")





# investigate thresholding image with various thresholds----------------------
threshold = 210
resmaps_val_uint8_th = threshold_images(resmaps_val_uint8, threshold) # resmaps_val_th_uint8
index_val = 0
plot_img(resmaps_val_uint8_th[index_val], title="resmaps_val_uint8_th[{}]\n threshold = {}".format(index_val, threshold))


threshold = 151
resmaps_test_uint8_th = threshold_images(resmaps_test_uint8, threshold)
index_test = 12
plot_img(resmaps_test_uint8_th[index_test], title="resmaps_test_uint8_th[{}]\n threshold = {}".format(index_test, threshold))


#================= AREA SIZE DISTRIBUTION FOR ALL THRESHOLDS ==================

def get_max_area(resmaps, threshold):
    resmaps_th = threshold_images(resmaps, threshold)
    # resmaps_fil = filter_median_images(resmaps_th)
    resmaps_labeled, areas_all = label_images(resmaps_th)
    areas_all_1d = [item for sublist in areas_all for item in sublist]
    max_area = np.amax(np.array(areas_all_1d))    
    return max_area

# def get_threshold_min(resmaps, ppf_value=0.97):
#     # compute descriptive values
#     minimum = np.amin(resmaps_val_uint8)
#     maximum = np.amax(resmaps_val_uint8)
#     mu = resmaps_val_uint8.flatten().mean()
#     sigma = resmaps_val_uint8.flatten().std()
#     th_min = int(round(scipy.stats.norm(mu, sigma).ppf(0.97), 1)) - 1
#     print("th_min = {} \n mu = {} \n sigma = {} \n min = {}\n max = {}".format(th_min, mu, sigma, minimum, maximum))
#     return th_min

def get_xbars(edges):
    x_bars = edges[:-1] + ((edges[1] - edges[0]) / 2)
    return x_bars

    
def get_area_size_counts_multiple_thresholds(resmaps_val_uint8, th_min=128, th_max=255, binning=True, nbins=200):
    if binning == True:
        # compute max_area
        max_area = get_max_area(resmaps_val_uint8, th_min)
        
        # initialize variables and parameters
        thresholds = np.arange(th_min, th_max+1)
        range_area = (0.5 , float(max_area)+0.5)
        counts = np.zeros(shape=(len(thresholds), nbins))
        
        # loop over all thresholds and calculate area counts (with binning)
        for i, threshold in enumerate(thresholds):
            resmaps_val_uint8_th = threshold_images(resmaps_val_uint8, threshold)
            # resmaps_val_uint8_th_fil = filter_median_images(resmaps_val_uint8_th)
            resmaps_val_labeled, areas_all_val = label_images(resmaps_val_uint8_th)
            areas_all_val_1d = [item for sublist in areas_all_val for item in sublist]
            count, edges = np.histogram(
                        areas_all_val_1d, 
                        bins=nbins, 
                        density=False, 
                        range=range_area
                    )
            counts[i] = count
        return counts, edges
            
    else:
        max_area = get_max_area(resmaps_val_uint8, th_min)
        # initialize variables and parameters
        thresholds = np.arange(th_min, th_max+1)
        areas = np.arange(1, max_area+1)
        counts = np.zeros(shape=(len(thresholds), max_area))
        
        # loop over all thresholds and calculate area counts
        for i, threshold in enumerate(thresholds):
            resmaps_val_uint8_th = threshold_images(resmaps_val_uint8, threshold)
            resmaps_val_uint8_th_fil = filter_median_images(resmaps_val_uint8_th)
            resmaps_val_labeled, areas_all_val = label_images(resmaps_val_uint8_th_fil)
            areas_all_val_1d = [item for sublist in areas_all_val for item in sublist]
            counts[i] = np.array([areas_all_val_1d.count(area) for area in areas])
        
        return counts, areas
    

def get_sum_counts(counts, edges, th_min, th_max, binning=True, plot=False):
    if binning == True:
        counts_sum = np.sum(counts, axis=0)
        x_bars = get_xbars(edges)    
        if plot:
            plt.figure()        
            plt.bar(x_bars, counts_sum, width=edges[1]-edges[0])
            plt.grid(which="both")
            plt.xlabel("areas_size")
            plt.ylabel("counts_sum (bins)")
            plt.title("sum of bin counts for thresholds in [{}, {}]".format(th_min, th_max))
            plt.show()
        return counts_sum, x_bars
    else:
        counts_sum = np.sum(counts, axis=0)
        if plot:
            plt.figure()
            plt.plot(edges, counts_sum)
            # plt.bar(areas, counts_sum) # TRY !!!
            plt.xlabel("areas_size")
            plt.ylabel("counts_sum")
            plt.title("sum of counts for thresholds in [{}, {}]".format(th_min, th_max))
            plt.show()
        return counts_sum, edges
        
            
    

# calculate sum of area counts (WITH binning) over all thresholds and plot
th_min = 128 
th_max = 255 # when nb_regions == 0, maximum 
counts, edges = get_area_size_counts_multiple_thresholds(resmaps_val_uint8, th_min, th_max, binning=True, nbins=200)
counts_sum, x_bars = get_sum_counts(counts, edges, th_min, th_max, binning=True, plot=True)

# calculate sum of area counts (WITHOUT binning) over all thresholds and plot
counts, areas = get_area_size_counts_multiple_thresholds(resmaps_val_uint8, th_min, th_max, binning=False)
counts_sum, areas = get_sum_counts(counts, areas, th_min, th_max, binning=False, plot=True)


#============== AREA SIZE DISTRIBUTION FOR SELECTED THRESHOLDS ================

# VISUALIZING
def plot_area_distro_for_multiple_thresholds(resmaps, thresholds_to_plot, th_min, nbins=200, method="line", title="provide a title"):    
    # fix range so that counts have the save x value
    max_area = get_max_area(resmaps, th_min)
    range_area = (0.5 , float(max_area)+0.5)
    
    # compute residual maps for multiple thresholds
    fig = plt.figure(figsize=(12, 5))
    for threshold in thresholds_to_plot:
        resmaps_th = threshold_images(resmaps, threshold)
        # resmaps_fil = filter_median_images(resmaps_th, kernel_size=3)
        resmaps_labeled, areas_all_val = label_images(resmaps_th)
        areas_all_1d = [item for sublist in areas_all_val for item in sublist]       
        
        if method == "hist":            
            count, bins, ignored = plt.hist(
                areas_all_1d,
                bins=nbins,
                density=False,
                range=range_area, # previously commented
                histtype="barstacked",
                label="threshold = {}".format(threshold),
            )
        
        elif method == "line":
            count, bins = np.histogram(
                areas_all_1d, 
                bins=nbins, 
                density=False, 
                range=range_area # previously commented
            )
            bins_middle = bins[:-1] + ((bins[1] - bins[0]) / 2)
            plt.plot(
                bins_middle,
                count,
                linestyle="-",
                linewidth=0.5, # 0.5
                marker="x", # "o"
                markersize=0.5, # 0.5
                label="threshold = {}".format(threshold),
            )
            # plt.fill_between(bins_middle, count)
    
    plt.title(title)
    plt.legend()
    plt.xlabel("area size in pixel")
    plt.ylabel("count")
    plt.grid()
    plt.show()
    # return bins, count, areas_all_1d


    
th_min = 128
    
thresholds_to_plot = [132, 137, 141] # [129, 135, 137, 139, 141]
title_val = "Distribution of anomaly areas' sizes for validation ResMaps with various Thresholds"
plot_area_distro_for_multiple_thresholds(resmaps_val_uint8, thresholds_to_plot, th_min, nbins=100, method="line", title=title_val)

resmaps_val_uint8_blur = filter_gauss_images(resmaps_val_uint8)
plot_area_distro_for_multiple_thresholds(resmaps_val_uint8_blur, thresholds_to_plot, 129, nbins=100, method="line", title="blur")

thresholds_to_plot = [132, 137, 141] # [129, 135, 137, 139, 141]
title_test = "Distribution of anomaly areas' sizes for test ResMaps with various Thresholds"
plot_area_distro_for_multiple_thresholds(resmaps_test_uint8, thresholds_to_plot, th_min, nbins=400, method="hist", title=title_test)





# ============= GET BASIC STATISTICS ON ANOMALOUD REGIONS =====================

def get_stats(resmaps, th_min=128, th_max=255, plot=False):
    
    dict_stat = {
        "threshold": [],
        "nb_regions": [],
        "mean_areas_size": [],
        "std_areas_size": [],
        "sum_areas_size": [],
    }

    # compute and plot number of anomalous regions and their area sizes with increasing thresholds
    print("computing anomalous regions and area sizes with increasing thresholds...")
    for threshold in range(th_min, th_max + 1):
        # threshold residual maps
        resmaps_th = threshold_images(resmaps, threshold)

        # filter images to remove salt noise
        # resmaps_fil = filter_median_images(resmaps_th, kernel_size=3)

        # compute anomalous regions and their size for current threshold
        resmaps_labeled, areas_all = label_images(resmaps_th)
        areas_all_1d = [item for sublist in areas_all for item in sublist]

        # compute the size of the biggest anomalous region (corresponds with smallest threshold)
        if threshold == th_min:
            max_region_size = np.amax(np.array(areas_all_1d))

        nb_regions = len(areas_all_1d)
        if nb_regions == 0:
            break

        mean_areas_size = np.mean(areas_all_1d)
        std_areas_size = np.std(areas_all_1d)
        sum_areas_size = np.sum(areas_all_1d)

        # append values to dictionnary
        dict_stat["threshold"].append(threshold)
        dict_stat["nb_regions"].append(nb_regions)
        dict_stat["mean_areas_size"].append(mean_areas_size)
        dict_stat["std_areas_size"].append(std_areas_size)
        dict_stat["sum_areas_size"].append(sum_areas_size)
        
    df_stat = pd.DataFrame.from_dict(dict_stat)
    
    if plot:
        fig = plt.figure()
        plt.style.use("seaborn-darkgrid")
        ax1 = fig.add_subplot(111)
        lns1 = ax1.plot(df_stat.threshold, df_stat.mean_areas_size, "C0", label="mean_areas_size") #1f77b4
        lns2 = ax1.plot(df_stat.threshold, df_stat.std_areas_size, "C1", label="std_areas_size")
        ax1.set_xlabel("Thresholds")
        ax1.set_ylabel("areas size [number of pixels]")
    
        ax2 = ax1.twinx()
        lns3 = ax2.plot(df_stat.threshold, df_stat.nb_regions, "C2", label="nb_regions") #ff7f0e
        ax2.set_ylabel("number of anomalous regions", color="C2")
        for tl in ax2.get_yticklabels():
            tl.set_color("C2")
        
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        
        plt.show()
    return df_stat
    
    
df_stat = get_stats(resmaps_val_uint8, th_min=128, th_max=255, plot=True)





# img = plt.imread("werkstueck/data_a30_nikon_weiss_edit/train/good/a30_nikon_weiss_train_good_fuss_007_edit.png")
# img2 = cv2.imread("werkstueck/data_a30_nikon_weiss_edit/train/good/a30_nikon_weiss_train_good_fuss_007_edit.png")
# plt.imshow()





