# -*- coding: utf-8 -*-

"""
This script is published in conjunction with Allison et al., 2025:
200 Ma of GDGT Uniformitarianism. JOURNAL TBC. DOI Link TBC
Code and README housed at:
https://github.com/matthew_allison05/OPTiMAL_2024

@author: Matthew Allison - University of Birmingham

This Python file contains the pre-written functions which provide the
code/analysis for Allison et al., 2025 - 200 Ma of GDGT Uniformitarianism.

Hopefully (famous last words...) this script will not be interacted with. 
The main script "OPTiMAL_Python.py" call generalised functions from here.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib import gridspec
import statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
import pickle
from pathlib import Path
from joblib import dump, load
from skmisc.loess import loess


def Return_Slice_of_Distance_df(df_distance, quartile, slice_option="Ancient"):
    # Inputs: 
    #   - df_distance: D_value matrix from MATLAB GP
    #   - quartile: Value used to slcie the D_value matrix (0 - 1)
    #   - slice_option: "Ancient" or "Calibration" to select which dataset
    #                   is being interrogated (rows vs columns)
    
    # Outputs: 
    #   - D_values: array of D_values sliced at given quartile     
    

    #print(f"Slicing df at quartile = {quartile} %....")
    
    # Get the array and the row and column count
    distance_array = df_distance.to_numpy()
    row, col = distance_array.shape

    
    # Slice for calibraiton data (by row)
    if slice_option == "Calibration":
        D_values = np.zeros(col)
        IQR_values = np.zeros(col)
        for n in range(0,(col)):
            data = distance_array[:,n]
            quant = np.nanquantile(data,quartile)
            pop_var = np.var(data)     
            # q25 = np.nanquantile(data, 0.25)
            # q75 = np.nanquantile(data, 0.75)
            # iqr = q75 - q25
            D_values[n] = quant
            IQR_values[n] = pop_var
    # Slice for ancient data (by col)        
    elif slice_option == "Ancient":
        D_values = np.zeros(row)
        IQR_values = np.zeros(row)
        for n in range(0,(row)):
            data = distance_array[n,:]
            quant = np.nanquantile(data,quartile)
            pop_var = np.var(data)     
            # q25 = np.nanquantile(data, 0.25)
            # q75 = np.nanquantile(data, 0.75)
            # iqr = q75 - q25
            D_values[n] = quant
            IQR_values[n] = pop_var
    
    return D_values, IQR_values

def Return_Given_Epoch_df(df_ancient, Epoch):
    # Inputs: 
    #   - df_ancient: dataframe containing many epochs worth of data. Needs a 
    #                   column header of "Epoch" to function.
    #   - Epoch: desired epoch to return
    
        
    # Outputs: 
    #   - df_ancient_epoch: df_ancient containing data with desired epoch.
    
    df_ancient_epoch = df_ancient[df_ancient['Epoch'].str.contains(f"{Epoch}")]
    
    return df_ancient_epoch

def Make_Calibration_Map(df_calibration, df_ancient, df_distance, cmin=2, cmax=5, vmin=0, diverge=3, quartile=0.5, size=50, ms=10, save_fig=False):
    # Inputs: 
    #   - df_calibration: Calibration data
    #   - df_ancient: Ancient data
    #   - df_distance: D_Value matrix
    #   - cmin: Minimum value of the colourmap. Below this, colours will clip (default yellow)
    #   - cmax: Maximum value of the colourmap. Above this, colours will clip (default Black)
    #   - diverge: Mid-point of the colourmap. You can change this to be any value
    #               in between cmin and cmax
    #   - quartile: The slice of the D_value matrix you wish to visualise. Is fed into
    #               function: Return_Slice_of_Distance_df
    #   - save_fig: True or False to save the figure after making it. 
    
    # These variables let you scale the fig up and down using the multiplier whilst 
    # maintaining the aspect ratio
    
    multiplier = 5
    fig_ratio_x = 3 * multiplier
    fig_ratio_y = 2 * multiplier
    
    fig, ax = plt.subplots(1,1,
                            subplot_kw={'projection': ccrs.Robinson()},
                            figsize=(fig_ratio_x,fig_ratio_y))
    
    fig.suptitle("D_value Calibration Map", fontsize = 30)

    # Sets up variables
    cmin_diverge = np.linspace(vmin,diverge,3)
    diverge_cmax = np.linspace(diverge,cmax,3)
    
    # Normalises the colourspace
    norm = mcolors.Normalize(vmin = cmin, vmax = cmax)
    
    # Custom colourmap options
    # Gets the real values
    col_1 = (cmin_diverge[1]/cmax)
    col_2 = (cmin_diverge[2]/cmax)
    col_3 = (diverge_cmax[1]/cmax)
    col_4 = (diverge_cmax[2]/cmax)
    
    # Gets the RGB setpoints
    vmin_RGB = [253/255, 225/255, 30/255]
    col_1_RGB = [247/255, 202/255, 96/255]
    col_2_RGB = [191/255, 54/255, 136/255]
    col_3_RGB = [104/255, 56/255, 168/255]
    col_4_RGB = [79/255, 17/255, 103/255]
    vmax_RGB = [30/255, 30/255, 30/255]
    
    # Creates a dict with all RGB combinations for the setpoints
    cdict = {'red':   ((vmin, vmin_RGB[0], vmin_RGB[0]),
                        (col_1, col_1_RGB[0], col_1_RGB[0]),
                        (col_2, col_2_RGB[0], col_2_RGB[0]),
                        (col_3, col_3_RGB[0], col_3_RGB[0]),
                        (col_4, col_4_RGB[0], col_4_RGB[0]),
                        (1.0, vmax_RGB[0], vmax_RGB[0])),
    
              'green': ((vmin, vmin_RGB[1], vmin_RGB[1]),
                        (col_1, col_1_RGB[1], col_1_RGB[1]),
                        (col_2, col_2_RGB[1], col_2_RGB[1]),
                        (col_3, col_3_RGB[1], col_3_RGB[1]),
                        (col_4, col_4_RGB[1], col_4_RGB[1]),
                        (1.0, vmax_RGB[1], vmax_RGB[1])),
    
              'blue':  ((vmin, vmin_RGB[2], vmin_RGB[2]),
                        (col_1, col_1_RGB[2], col_1_RGB[2]),
                        (col_2, col_2_RGB[2], col_2_RGB[2]),
                        (col_3, col_3_RGB[2], col_3_RGB[2]),
                        (col_4, col_4_RGB[2], col_4_RGB[2]),
                        (1.0, vmax_RGB[2], vmax_RGB[2]))}
    
    # Makes custon colourmap and normalised
    rgb_cmap = LinearSegmentedColormap('RGB', cdict)
    cmap_custom = plt.cm.ScalarMappable(cmap=rgb_cmap, norm=norm)

    # Adds global feature set
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", zorder=0)
    ax.add_feature(cfeature.BORDERS, edgecolor="darkgrey", zorder=0)
    ax.gridlines(zorder=0)
    ax.set_facecolor('lightgrey')
    
    # Gets slice of D_Value array fromt he ancient and modern data
    quartile = quartile
    df_distance = df_distance
    CD_values, CD_IQR = Return_Slice_of_Distance_df(df_distance,quartile,slice_option="Ancient")
    df_ancient["D_Values"] = CD_values

    CD_values, CD_IQR = Return_Slice_of_Distance_df(df_distance,quartile,slice_option="Calibration")
    df_calibration["D_Values"] = CD_values
    
    # Get the lat lon data of the calibration data
    lon = df_calibration["Longitude"].to_numpy()
    lat = df_calibration["Latitude"].to_numpy()
    c = CD_values

    # Collect data ready for plotting
    xyc = np.stack((lat,lon,c), axis=-1)
    df_plot = pd.DataFrame(xyc, columns=("x","y","colour"))
    df_plot = df_plot.sort_values(by='colour', ascending=False)
    xyc = df_plot.to_numpy()

    # Plot the OPTiMAL output data
    ax.scatter(xyc[:,1] ,xyc[:,0] , c=xyc[:,2], cmap = rgb_cmap, norm=norm, s=size,
               transform =ccrs.PlateCarree(), label="OPTiMAL Data")

    # Collect the sample site lat lon
    lon = df_calibration["Longitude"].to_numpy()
    lat = df_calibration["Latitude"].to_numpy()
    # Plot drill sites of fossil data on the plot
    drill_site_lat = df_ancient["Latitude"].to_numpy()
    drill_site_lon = df_ancient["Longitude"].to_numpy()

    # Plot the ancient data sampling site
    ax.plot(drill_site_lon ,drill_site_lat, marker = 'P', ms=ms, 
            markerfacecolor="limegreen", markeredgecolor='black', markeredgewidth=0.5, linestyle='None',
            transform =ccrs.PlateCarree(), label="Sample Site")
    
    # Plot colourbar
    cbar = fig.colorbar(cmap_custom, cmap = rgb_cmap, ax=ax, location = 'bottom', orientation='horizontal', shrink=0.75)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='D_Value', size=20)
    
    # Plot legend and show figure
    plt.legend()
    plt.tight_layout()  
    
    # If save_fig is True, save the figure as a png and svg file
    if save_fig == True:
        plt.savefig("Global_Calibration_Plot.svg") 
        plt.savefig("Global_Calibration_Plot.png")
    else:
        pass
    
    plt.show()    
    
def OPTiMAL_SST_Timeseries(df_calibration, df_ancient, see_QC_Fail=True, see_epochs=True, save_fig=False):    
    # Inputs: 
    #   - df_calibration: Calibration data
    #   - df_ancient: Ancient data
    #   - see_QC_Fail: True or False. Adds the striped background to the plot
    #                   showing where data of various quality exists.
    #   - see_Epoch: True or False. Adds coloured bars to the bottom of the plot
    #                   The colour of the bars and the sizes correlate to the
    #                   ICC chronostrat chart
    #   - save_fig: True or False to save the figure after making it. 
    
    # These variables let you scale the fig up and down using the multiplier whilst 
    # maintaining the aspect ratio

    multiplier = 7
    fig_ratio_x = 5 * multiplier
    fig_ratio_y = 2 * multiplier
    
    fig, ax = plt.subplots(1,1,
                           figsize=(fig_ratio_x,fig_ratio_y))
       
    fig.suptitle("OPTiMAL SST Timeseries", fontsize = 30)
    
    # Add the generally useful Palaeolatitude absolute column
    df_ancient["Palaeolatitude_abs"] = abs(df_ancient["Palaeolatitude"])
    
    # Separate out pass and fails using D_Nearest <= 0.5
    df_ancient_pass = df_ancient[df_ancient["D_Nearest"] <= 0.5]
    df_ancient_fail = df_ancient[df_ancient["D_Nearest"] > 0.5]

    if see_QC_Fail == True:
        # Generate the data QC stripes using a histogram
        # Get x data
        t_max = max(df_ancient["Age"].to_numpy())
        t_min = min(df_ancient["Age"].to_numpy())
        
        x_pass = df_ancient_pass["Age"]
        x_fail = df_ancient_fail["Age"]
        
        # Creat the x axis bins at widths of 0.25 Ma or 20 per timeseries,
        # whatever is smallest
        stripe_width = 0.25/(t_max - t_min)
        t_range = (t_max - t_min) * 4
        print(t_range, "Range")
        if t_range > 20:
            x_max = t_max + 0.25
            x_min = t_min - 0.25
            xbins = np.linspace(x_min,x_max,int(t_range))
        else:
            x_max = t_max + (t_max/10)
            x_min = t_min - (t_max/10)
            xbins = np.linspace(x_min,x_max,20)   
        
        # xbins = np.linspace(0,200,800)
        # Get the fail data
        df_choice = df_ancient_fail
        x = df_choice["Age"]
        # Bin x data for the dark grey (good data present) stripes
        count_centers, bins  = np.histogram(x, xbins)
        center = (bins[:-1] + bins[1:]) / 2
   
        # Bin x data
        count_centers = count_centers.astype('float')
        for i in range(0,len(count_centers)):
            if count_centers[i] != 0:
               count_centers[i] = 1.0
            else:
                count_centers[i] = np.nan
            
        count_bins = np.zeros(((len(count_centers)+1)))    
        for i in range(0,len(count_centers)):
            if np.isnan(count_centers[i]) == True:
                count_bins[i] = np.nan
                count_bins[i+1] = np.nan
            else:
                pass 
            
        for i in range(0,len(count_centers)):
            if count_centers[i] == 1.0:
                count_bins[i] = 1.0
                count_bins[i+1] = 1.0
            else:
                pass             
                
        df_bins = pd.DataFrame([bins, count_bins]).T
        df_bins = df_bins
        df_center = pd.DataFrame([center, count_centers])
        df_center = df_center.T
   
        df_stripe = pd.concat([df_bins, df_center])
        df_stripe = df_stripe.sort_values(df_stripe.columns[0], ascending=True)
           
        # ax.scatter(x,y,marker='D')
        ax.fill_between(df_stripe[0],df_stripe[1],-90,color='lightgrey')
        ax.fill_between(df_stripe[0],df_stripe[1],90,color='lightgrey', label="Data exists but fails QC")
   
        # Get the fail data
        df_choice = df_ancient_pass
        x = df_choice["Age"]
        # Bin x data for the dark grey (good data present) stripes
        count_centers, bins  = np.histogram(x, xbins)
        center = (bins[:-1] + bins[1:]) / 2
   
        # Bin x data
        count_centers = count_centers.astype('float')
        for i in range(0,len(count_centers)):
            if count_centers[i] != 0:
                count_centers[i] = 1.0
            else:
                count_centers[i] = np.nan
            
        count_bins = np.zeros(((len(count_centers)+1)))    
        for i in range(0,len(count_centers)):
            if np.isnan(count_centers[i]) == True:
                count_bins[i] = np.nan
                count_bins[i+1] = np.nan
            else:
                pass 
            
        for i in range(0,len(count_centers)):
            if count_centers[i] == 1.0:
                count_bins[i] = 1.0
                count_bins[i+1] = 1.0
            else:
                pass             
               
        df_bins = pd.DataFrame([bins, count_bins]).T
        df_bins = df_bins
        df_center = pd.DataFrame([center, count_centers])
        df_center = df_center.T
   
        df_stripe = pd.concat([df_bins, df_center])
        df_stripe = df_stripe.sort_values(df_stripe.columns[0], ascending=True)
           
        # ax.scatter(x,y,marker='D')
        ax.fill_between(df_stripe[0],df_stripe[1],-90,color='darkgrey')
        ax.fill_between(df_stripe[0],df_stripe[1],90,color='darkgrey', label="Data exists and pass QC") 
        
    else:
        pass
    
    if see_epochs == True:
        # Adds a visual aid of Epochs along the bottom of the palaeolatitude plot
        x_box = -5
        x_thickness = 2
        # Pleistocene
        rect = patches.Rectangle((0,x_box),2.5,x_thickness,facecolor='wheat',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Pliocene
        rect = patches.Rectangle((2.5,x_box),2.8,x_thickness,facecolor='gold',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Miocene
        rect = patches.Rectangle((5.3,x_box),17.7,x_thickness,facecolor='yellow',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Oligocene
        rect = patches.Rectangle((23,x_box),10.9,x_thickness,facecolor='sandybrown',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Eocene
        rect = patches.Rectangle((33.9,x_box),22.1,x_thickness,facecolor='coral',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Palaeocene
        rect = patches.Rectangle((56.0,x_box),10,x_thickness,facecolor='tomato',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Cretaceous
        rect = patches.Rectangle((66.0,x_box),79,x_thickness,facecolor='limegreen',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Jurassic
        rect = patches.Rectangle((145,x_box),55,x_thickness,facecolor='cornflowerblue',edgecolor='black',zorder=4)
        ax.add_patch(rect)
    else: 
        pass
    
    # Axis 1: Plots failed data points (X)
    df_choice = df_ancient_fail
    x = df_choice["Age"]
    y = df_choice["Palaeolatitude_abs"]
    c = df_choice["Predicted_Temp"]

    ax.scatter(x,y, c='black',marker='x', label="D_nearest > 0.5",s=150, alpha=0.8, zorder = 2)

    # Axis 1: Plots pass data points (O) coloured by OPTiMAL temp predictions
    df_choice = df_ancient_pass
    x = df_choice["Age"]
    y = df_choice["Palaeolatitude_abs"]
    c = df_choice["Predicted_Temp"]
      
    im = ax.scatter(x,y,c=c,cmap='RdYlBu_r',s=400, label="Data", zorder = 2)
    # Sets Axis lims
    ax.set_ylim((-5,90))
    ax.set_xlim((min(df_ancient["Age"]),max(df_ancient["Age"])))
    ax.set_xscale('linear')

    cbaxes = ax.inset_axes([1,0,0.025,1]) 
    plt.colorbar(im, cax=cbaxes, label="OPTiMAL SST [^oC]")
    
    plt.xlabel("Age [Ma]")
    plt.ylabel("Absolute Palaeolatitude [Degrees]")
   
    ax.legend()
    
    # If save_fig is True, save the figure as a png and svg file
    if save_fig == True:
        plt.savefig("OPTiMAL_SST_Timeseries.svg") 
        plt.savefig("OPTiMAL_SST_Timeseries.png")
    else:
        pass
   
    plt.show()            
     
    return
        
     
def OPTiMAL_CDvalue_Timeseries(df_calibration, df_ancient, df_distance, quartile = 0.5, see_epochs=True, save_fig=False):   
    # Inputs: 
    #   - df_calibration: Calibration data
    #   - df_ancient: Ancient data
    #   - df_distance: D_Value matrix
    #   - quartile: The slice of the D_value matrix you wish to visualise. Is fed into
    #               function: Return_Slice_of_Distance_df.
    #   - see_Epoch: True or False. Adds coloured bars to the bottom of the plot
    #                   The colour of the bars and the sizes correlate to the
    #                   ICC chronostrat chart
    #   - save_fig: True or False to save the figure after making it. 
    
    # These variables let you scale the fig up and down using the multiplier whilst 
    # maintaining the aspect ratio
    
    multiplier = 7
    fig_ratio_x = 5 * multiplier
    fig_ratio_y = 2 * multiplier
    
    fig, ax = plt.subplots(1,1,
                           figsize=(fig_ratio_x,fig_ratio_y))
       
    fig.suptitle("OPTiMAL D_Values Timeseries", fontsize = 30)
    
    # Gets slice of D_Value array fromt he ancient and modern data
    # D_values = Return_Slice_of_Distance_df(df_distance,quartile,slice_option="Ancient")
    # df_ancient["D_Values"] = D_values
    CD_values, CD_IQR = Return_Slice_of_Distance_df(df_distance,0.5,slice_option="Ancient")
    df_ancient["D_Values"] = CD_values
    df_ancient["CD_IQR"] = CD_IQR
    
    # Add the generally useful Palaeolatitude absolute column
    df_ancient["Palaeolatitude_abs"] = abs(df_ancient["Palaeolatitude"])
    
    # Trim the D_Values to remove outrageous outliers
    trimmer = 14
    df_ancient = df_ancient[df_ancient["D_Values"] < trimmer]
    
    # Preps a small df and runs the loess regression
    def prep_df(df, x='x', y='y', var='var'):
        """
        Return a clean slice with columns: x, y, sigma
        - Coerces to numeric
        - Drops NaN/inf
        - Requires sigma > 0
        - Sorts by x
        """
        out = df[[x, y, var]].copy()
        out.columns = ['x', 'y', 'var']
        # make numeric & clean
        for c in ['x','y','var']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        out = (out
               .replace([np.inf, -np.inf], np.nan)
               .dropna(subset=['x','y','var']))
        out = out[out['var'] > 0]
        return out.sort_values('x').reset_index(drop=True)
    
    out = prep_df(df_ancient, x='Age', y='D_Values', var='CD_IQR')
    
    x = out["x"].to_numpy()
    y = out["y"].to_numpy()
    w = out["var"].to_numpy()
    w = 1/w
    
    l = loess(x, y, weights=w, span=0.0075, degree=2, family="symmetric")
    l.fit()
    
    x_new = np.linspace(x.min(), x.max(), 800)
    pred = l.predict(x_new, stderror=True)
    
    y_hat = pred.values           # fitted curve
    y_se  = pred.stderr           # pointwise SE
    ci    = pred.confidence(0.05) # 95% CI
    lo, hi = ci.lower, ci.upper
        
    # Separate out pass and fails using D_Nearest <= 0.5
    df_ancient_pass = df_ancient[df_ancient["D_Nearest"] <= 0.5]
    df_ancient_fail = df_ancient[df_ancient["D_Nearest"] > 0.5]
    
    x_temp = df_ancient_fail["Age"].to_numpy()
    y_temp = df_ancient_fail["D_Values"].to_numpy()
    ax.scatter(x_temp,y_temp,marker = 'x', c = 'gray', s=50,
                label="OPTiMAL Fail, D_Nearest > 0.5", alpha = 0.5, zorder = 1) 
    
    x_temp = df_ancient_pass["Age"].to_numpy()
    y_temp = df_ancient_pass["D_Values"].to_numpy()
    # ax.scatter(x_temp,y_temp,marker = 'o', c = 'cornflowerblue',# edgecolors='black',
    #             linewidth=0.5 ,s = 150, label="OPTiMAL Pass, D_Nearest <= 0.5", edgecolor='black', alpha = 0.75, zorder = 1)  
    # ax.scatter(x_temp,y_temp,marker = 'o', c = 'cornflowerblue', s=50,
    #             label="OPTiMAL Pass, D_Nearest <= 0.5", alpha = 0.5, zorder = 1) 
    ax.scatter(x_temp,y_temp,marker = 'o', c = 'cornflowerblue', s=50,
                label="OPTiMAL Fail, D_Nearest > 0.5", alpha = 0.6, zorder = 1)  
    
    ax.plot(x_new, pred, 'r-', linewidth=5, label='GPR mean', zorder=5)
    # y_pred, y_std = gpr.predict(x_pred, return_std=True)
    ax.fill_between(x_new, lo, hi, color='coral', alpha=0.7, label='LOESS 95th% band', zorder=4)
    ax.set_ylim((0,15))
       
    if see_epochs == True:
        # Adds a visual aid of Epochs along the bottom of the palaeolatitude plot
        x_box = 0
        x_thickness = 0.25
        # Pleistocene
        rect = patches.Rectangle((0,x_box),2.5,x_thickness,facecolor='wheat',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Pliocene
        rect = patches.Rectangle((2.5,x_box),2.8,x_thickness,facecolor='gold',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Miocene
        rect = patches.Rectangle((5.3,x_box),17.7,x_thickness,facecolor='yellow',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Oligocene
        rect = patches.Rectangle((23,x_box),10.9,x_thickness,facecolor='sandybrown',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Eocene
        rect = patches.Rectangle((33.9,x_box),22.1,x_thickness,facecolor='coral',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Palaeocene
        rect = patches.Rectangle((56.0,x_box),10,x_thickness,facecolor='tomato',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Cretaceous
        rect = patches.Rectangle((66.0,x_box),79,x_thickness,facecolor='limegreen',edgecolor='black',zorder=4)
        ax.add_patch(rect)
        # Jurassic
        rect = patches.Rectangle((145,x_box),55,x_thickness,facecolor='cornflowerblue',edgecolor='black',zorder=4)
        ax.add_patch(rect)
    else: 
        pass
    
    ax.set_xlim((min(df_ancient["Age"]),max(df_ancient["Age"])))
    ax.set_xscale('linear')
    ax.set_ylim(0, trimmer)
    
    plt.xlabel("Age [Ma]")
    plt.ylabel("OPTiMAL CD_Median Values")
   
    ax.legend()
    
        
    # If save_fig is True, save the figure as a png and svg file
    if save_fig == True:
        plt.savefig("OPTiMAL_CD_Values_Timeseries.svg") 
        plt.savefig("OPTiMAL_CD_Values_Timeseries.png")
    else:
        pass
    
    plt.tight_layout()  
    plt.show()
    
    return

def ODP_1168_1172(df_calibration, df_ancient, df_distance, quartile = 0.5, see_epochs=True, save_fig=False):  
    # Inputs: 
    #   - df_calibration: Calibration data
    #   - df_ancient: Ancient data
    #   - df_distance: D_Value matrix
    #   - quartile: The slice of the D_value matrix you wish to visualise. Is fed into
    #               function: Return_Slice_of_Distance_df.
    #   - see_Epoch: True or False. Adds coloured bars to the bottom of the plot
    #                   The colour of the bars and the sizes correlate to the
    #                   ICC chronostrat chart
    #   - save_fig: True or False to save the figure after making it. 
    
    # These variables let you scale the fig up and down using the multiplier whilst 
    # maintaining the aspect ratio
    multiplier = 10
    fig_ratio_x = 2 * multiplier
    fig_ratio_y = 2 * multiplier
    
    fig, ax = plt.subplots(figsize=(fig_ratio_x,fig_ratio_y))
    
    # 4 Axes to make
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1]) 
    # TEX86
    ax1 = plt.subplot(gs[0])
    # TEX86 derived SST (Twin of ax1)
    ax2 = ax1.twinx()
    # OPTiMAL D_Values
    ax3 = plt.subplot(gs[1],sharex=ax1)
    # OPTiMAL SST
    ax4 = plt.subplot(gs[2],sharex=ax1)
       
    fig.suptitle("ODP 1168 and 1172", fontsize = 30)

    # Extract ODP 1168 and 1172 from the global database
    df_1168 = df_ancient[df_ancient["Site"] == "ODP 1168"]
    df_1172 = df_ancient[df_ancient["Site"] == "ODP 1172"]
    # Make a dataframe for just these ODP sites
    frames = [df_1168, df_1172]
    df_site_combo = pd.concat(frames)
    
    # Add the generally useful Palaeolatitude absolute column
    df_site_combo["Palaeolatitude_abs"] = abs(df_site_combo["Palaeolatitude"])
    
    # Trim the D_Values to remove outrageous outliers
    trimmer = 14
    df_site_combo = df_site_combo[df_site_combo["D_Values"] < trimmer]

    # Axis 1 and 2 ~~~~ TEX86 and TEX derived SST ~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Add Tex86 into the dataframe
    df_site_combo["Tex86"] = (df_site_combo["GDGT_2"]+df_site_combo["GDGT_3"]+df_site_combo["Cren"])/(df_site_combo["GDGT_1"]+df_site_combo["GDGT_2"]+df_site_combo["GDGT_3"]+df_site_combo["Cren"])
    # Add Tex86 derived SST into the dataframe
    df_site_combo["Tex_SST"] = (df_site_combo["Tex86"] - 0.28)/0.015
    
    # Set the alpha for transparency in plotting
    alpha = 0.5

    # Site 1168...
    
    # Extract ODP 1168 from the combined dataframe
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1168"]
    df_temp = df_temp.sort_values(by='Age', ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    # Get the D_Nearest fails and passes
    df_temp_fail = df_temp[df_temp["D_Nearest"] > 0.5]
    df_temp_pass = df_temp[df_temp["D_Nearest"] <= 0.5]

    # Get the temporary xy data for the D_Nearest failures
    x_temp = df_temp_fail["Age"]
    y_temp = df_temp_fail["Tex_SST"]
    
    # Plot the D_Nearest fails
    ax1.scatter(x_temp,y_temp,c='navy', s=50, marker= 'x',alpha=alpha,zorder=1,
                label = "OPD 1168 OPTiMAL Failure")

    # Get the temporary xy data for the D_Nearest passes
    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["Tex_SST"]
    y_temp_sec = df_temp_pass["Tex86"]
    
    # Plot the D_Nearest passes
    ax1.scatter(x_temp,y_temp, s=100, c='cornflowerblue', marker = 'o',alpha=0.8,zorder=1,
                label = "OPD 1168 OPTiMAL Pass")
    # This plots the TEX86 raw values on a secondary y-axis on the first subplot
    # This is a single call for only part of the dataset as I just need to 
    # establish the secondary axis relative to the primary SST axis
    ax2.scatter(x_temp,y_temp_sec, s=100, c='navy', marker = 'o',alpha=0.0,zorder=1)

    # Now need to plot the Loess regression through ODP 1168
    # Recollect the entire data (passes and fails)
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1168"]
    x_temp = df_temp["Age"]
    y_temp = df_temp["Tex_SST"]
    
    # Calculate the Loess regression
    # Frac = fraction of the data used within a given sub-sample
    smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=0.05)

    # Plot the Loess regression
    ax1.plot(smoothed[:, 0], smoothed[:, 1], c="k", linewidth=3,
                label = "Loess Regression")

    # Site 1172...
    
    # Extract ODP 1172 from the combined dataframe
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1172"]
    df_temp = df_temp.sort_values(by='Age', ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    # Get the D_Nearest fails and passes
    df_temp_fail = df_temp[df_temp["D_Nearest"] > 0.5]
    df_temp_pass = df_temp[df_temp["D_Nearest"] <= 0.5]

    # Get the temporary xy data for the D_Nearest failures
    x_temp = df_temp_fail["Age"]
    y_temp = df_temp_fail["Tex_SST"]

    # Plot the D_Nearest fails
    ax1.scatter(x_temp,y_temp,c='maroon', s=50, marker= 'x',alpha=alpha,zorder=1,
                label = "OPD 1172 OPTiMAL Failure")

    # Get the temporary xy data for the D_Nearest passes
    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["Tex_SST"]

    # Plot the D_Nearest passes
    ax1.scatter(x_temp,y_temp, s=100, c='tomato', marker = 'o',alpha=0.8,zorder=1,
                label = "OPD 1168 OPTiMAL Pass")

    # Now need to plot the Loess regression through ODP 1172
    # Recollect the entire data (passes and fails)
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1172"]
    x_temp = df_temp["Age"]
    y_temp = df_temp["Tex_SST"]

    # Calculate the Loess regression
    # Frac = fraction of the data used within a given sub-sample
    smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=0.05)

    # Plot the Loess regression
    ax1.plot(smoothed[:, 0], smoothed[:, 1], c="k", linewidth=3)

    # Limits for the TEX86 and TEX86 SST y-axes
    x_low = 0
    x_high = 40
    ax1.set_ylim((x_low,x_high))
    ax2.set_ylim((x_low + 0.28),(x_high*0.015)+0.28)
    
    ax1.legend()

    # Axis 3 ~~~~~~ OPTiMAL CD_Value Plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Much of the following code is identical in structure to the first subplot
    # so I will not re-comment. Temporary xy's and df's were used for this reason
    # The only difference is the substitution of D_values instead of TEX86 for
    # the y-axis variable.
    
    # Site 1168 fails and passes...
    
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1168"]
    df_temp = df_temp.sort_values(by='Age', ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    df_temp_fail = df_temp[df_temp["D_Nearest"] > 0.5]
    df_temp_pass = df_temp[df_temp["D_Nearest"] <= 0.5]

    x_temp = df_temp_fail["Age"]
    y_temp = df_temp_fail["D_Values"]

    ax3.scatter(x_temp,y_temp,c='navy', s=50, marker= 'x',alpha=alpha,zorder=1)
    
    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["D_Values"]

    ax3.scatter(x_temp,y_temp, s=100, c='cornflowerblue', marker = 'o',alpha=0.8,zorder=1)
 
    # Site 1172 fails and passes...
    
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1172"]
    df_temp = df_temp.sort_values(by='Age', ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    df_temp_fail = df_temp[df_temp["D_Nearest"] > 0.5]
    df_temp_pass = df_temp[df_temp["D_Nearest"] <= 0.5]

    x_temp = df_temp_fail["Age"]
    y_temp = df_temp_fail["D_Values"]

    ax3.scatter(x_temp,y_temp,c='maroon', s=50, marker= 'x',alpha=alpha,zorder=1)

    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["D_Values"]

    ax3.scatter(x_temp,y_temp, s=100, c='tomato', marker = 'o',alpha=0.8,zorder=1)

    # Site 1168 Loess regression...
    
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1168"]
    x_temp = df_temp["Age"]
    y_temp = df_temp["D_Values"]

    # Frac = fraction of the data used within a given sub-sample
    smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=0.05)

    ax3.plot(smoothed[:, 0], smoothed[:, 1], c="k", linewidth=3)

    # Site 1172 Loess regression...
    
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1172"]
    x_temp = df_temp["Age"]
    y_temp = df_temp["D_Values"]

    # Frac = fraction of the data used within a given sub-sample
    smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=0.05)

    ax3.plot(smoothed[:, 0], smoothed[:, 1], c="k", linewidth=3)
    
    # Axis 4 ~~~~~~~~~ OPTiMAL SST Plot ~~~~~~~~~~~~~~~~~~~~~~~
    
    # Now plot the D_Nearest pass data and the resulting OPTiMAL SST
    # predictions (with 95th% confidence intervals).
    
    # Establish new alpha for transparency (due to the fill_between) and
    # frac for the Loess regression (due to the sparcity of data).
    alpha = 0.5
    frac = 0.075
    
    # Site 1168...
    
    # Extract ODP 1168 from the combined dataframe
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1168"]
    df_temp = df_temp.sort_values(by='Age', ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    # Separate out passes and fails
    df_temp_fail = df_temp[df_temp["D_Nearest"] > 0.5]
    df_temp_pass = df_temp[df_temp["D_Nearest"] <= 0.5]

    x_temp = df_temp_fail["Age"]
    y_temp = df_temp_fail["Predicted_Temp"]

    # Comment out the failures to highlight/plot the passes only
    # ax4.scatter(x_temp,y_temp,c='navy', s=50, marker= 'x',alpha=alpha,zorder=1)

    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["Predicted_Temp"]

    # Plot the pass data
    ax4.scatter(x_temp,y_temp, s=100, c='cornflowerblue', marker = 'o',alpha=0.8,zorder=1)

    # Collect the 95th % data
    df_temp = df_temp
    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["Predicted_Temp"]
    y_upp_95 = df_temp_pass["Temp_upp_95"]
    y_low_95 = df_temp_pass["Temp_low 95"]

    # Thruple of Loess regressions in order to plot the data and both 95th
    # confidence intervals.
    # Frac = fraction of the data used within a given sub-sample
    smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=frac)

    ax4.plot(smoothed[:, 0], smoothed[:, 1], c="k", linewidth=3)

    # Frac = fraction of the data used within a given sub-sample
    smoothed_up = sm.nonparametric.lowess(exog=x_temp, endog=y_upp_95, frac=frac)
    ax4.plot(smoothed_up[:, 0], smoothed_up[:, 1], c="k", linewidth=1)
    # Frac = fraction of the data used within a given sub-sample
    smoothed_low = sm.nonparametric.lowess(exog=x_temp, endog=y_low_95, frac=frac)
    ax4.plot(smoothed_low[:, 0], smoothed_low[:, 1], c="k", linewidth=1)

    ax4.fill_between(smoothed[:, 0],smoothed_up[:, 1],smoothed_low[:, 1], color='cornflowerblue', alpha=0.3, zorder=0)

    # Site 1172...
    
    # Extract ODP 1168 from the combined dataframe
    df_temp = df_site_combo[df_site_combo["Site"] == "ODP 1172"]
    df_temp = df_temp.sort_values(by='Age', ascending=True)
    df_temp = df_temp.reset_index(drop=True)

    # Separate out passes and fails
    df_temp_fail = df_temp[df_temp["D_Nearest"] > 0.5]
    df_temp_pass = df_temp[df_temp["D_Nearest"] <= 0.5]

    x_temp = df_temp_fail["Age"]
    y_temp = df_temp_fail["Predicted_Temp"]

    # Comment out the failures to highlight/plot the passes only
    # ax4.scatter(x_temp,y_temp,c='maroon', s=50, marker= 'x',alpha=alpha,zorder=1)

    x_temp = df_temp_pass["Age"] 
    y_temp = df_temp_pass["Predicted_Temp"]

    # Plot the passes
    ax4.scatter(x_temp,y_temp, s=100, c='tomato', marker = 'o',alpha=0.8,zorder=1)

    # Collect the 95th % data
    df_temp = df_temp
    x_temp = df_temp_pass["Age"]
    y_temp = df_temp_pass["Predicted_Temp"]
    y_upp_95 = df_temp_pass["Temp_upp_95"]
    y_low_95 = df_temp_pass["Temp_low 95"]

    # Thruple of Loess regressions in order to plot the data and both 95th
    # confidence intervals.
    # Frac = fraction of the data used within a given sub-sample
    smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=frac)

    ax4.plot(smoothed[:, 0], smoothed[:, 1], c="k", linewidth=3,
                label = "Loess Regression")

    # Frac = fraction of the data used within a given sub-sample
    smoothed_up = sm.nonparametric.lowess(exog=x_temp, endog=y_upp_95, frac=frac)
    ax4.plot(smoothed_up[:, 0], smoothed_up[:, 1], c="k", linewidth=1,
                label = "95th Percentile Confidence Interval")
    # Frac = fraction of the data used within a given sub-sample
    smoothed_low = sm.nonparametric.lowess(exog=x_temp, endog=y_low_95, frac=frac)
    ax4.plot(smoothed_low[:, 0], smoothed_low[:, 1], c="k", linewidth=1)

    ax4.fill_between(smoothed[:, 0],smoothed_up[:, 1],smoothed_low[:, 1], color='tomato', alpha=0.3, zorder=0)
    
    ax4.legend()
    
    if see_epochs == True:
        # Adds a visual aid of Epochs along the bottom of the palaeolatitude plot
        x_box = 0
        x_thickness = 1.5
        # Pleistocene
        rect = patches.Rectangle((0,x_box),2.5,x_thickness,facecolor='wheat',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Pliocene
        rect = patches.Rectangle((2.5,x_box),2.8,x_thickness,facecolor='gold',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Miocene
        rect = patches.Rectangle((5.3,x_box),17.7,x_thickness,facecolor='yellow',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Oligocene
        rect = patches.Rectangle((23,x_box),10.9,x_thickness,facecolor='sandybrown',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Eocene
        rect = patches.Rectangle((33.9,x_box),22.1,x_thickness,facecolor='coral',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Palaeocene
        rect = patches.Rectangle((56.0,x_box),10,x_thickness,facecolor='tomato',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Cretaceous
        rect = patches.Rectangle((66.0,x_box),79,x_thickness,facecolor='limegreen',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
        # Jurassic
        rect = patches.Rectangle((145,x_box),55,x_thickness,facecolor='cornflowerblue',edgecolor='black',zorder=4)
        ax4.add_patch(rect)
    else: 
        pass
    
    # Housekeeping for axis labels
    ax1.set_xlabel("Age [Ma]")
    ax3.set_xlabel("Age [Ma]")
    ax4.set_xlabel("Age [Ma]")
    
    ax1.set_ylabel("TEX86 Derived SST [oC]")
    ax2.set_ylabel("TEX86")
    ax3.set_ylabel("OPTiMAL CD_median")
    ax4.set_ylabel("OPTiMAL SST Predictions [oC]")
    
    ax1.set_xlim((min(df_site_combo["Age"]),max(df_site_combo["Age"])))
    ax3.set_ylim(0, trimmer)
    ax4.set_ylim((0,35))    
        
    # If save_fig is True, save the figure as a png and svg file
    if save_fig == True:
        plt.savefig("OPTiMAL_D_Values_Timeseries.svg") 
        plt.savefig("OPTiMAL_D_Values_Timeseries.png")
    else:
        pass
    
    plt.tight_layout()  
    plt.show()
    
    return

def Failure_Rates_Palaoelatitude(epoch, ancient_df):
    
    df = ancient_df
    
    epoch = epoch
    df_epoch = Return_Given_Epoch_df(ancient_df,epoch)
    
    df_pass = df_epoch[df_epoch["D_Nearest"] <= 0.5]
    df_fails = df_epoch[df_epoch["D_Nearest"] > 0.5]
    
    fig = plt.figure(figsize=(20,28))
    
    ax = fig.add_subplot((111), projection=ccrs.Robinson(), zorder=0)
    ax.set_global()
    ax.set_title(f"{epoch}")
    
    # ax.scatter(df_pass["Palaeolongitude"].to_numpy(),df_pass["Palaeolatitude"].to_numpy(),transform =ccrs.PlateCarree())
    
    df_pass_x = np.cos(np.deg2rad(df_pass["Palaeolatitude"].to_numpy()))
    df_pass_x = np.rad2deg(df_pass_x) + 90
    
    s = 250
    
    ax.scatter(df_pass_x,(df_pass["Palaeolatitude"]).to_numpy(), marker = "o", s=s, c='cornflowerblue', edgecolors ='black',transform =ccrs.PlateCarree())
    
    df_fails_x = np.cos(np.deg2rad(df_fails["Palaeolatitude"].to_numpy()))
    df_fails_x = np.rad2deg(df_fails_x) + 110
    
    ax.scatter(df_fails_x,(df_fails["Palaeolatitude"]).to_numpy(), marker = "o", s=s, c='grey', edgecolors ='black',transform =ccrs.PlateCarree())

    ax.plot(np.array([-180,180]), np.array([60,60]), c = "gray", transform =ccrs.PlateCarree(), zorder = 0)
    ax.plot(np.array([-180,180]), np.array([30,30]), c = "gray", transform =ccrs.PlateCarree(), zorder = 0)
    ax.plot(np.array([-180,180]), np.array([0,0]), c = "gray", transform =ccrs.PlateCarree(), zorder = 0)
    ax.plot(np.array([-180,180]), np.array([-60,-60]), c = "gray", transform =ccrs.PlateCarree(), zorder = 0)
    ax.plot(np.array([-180,180]), np.array([-30,-30]), c = "gray", transform =ccrs.PlateCarree(), zorder = 0)
    
    plt.savefig(f"Failure_Palaeolatitude_{epoch}.svg") 
    
    plt.show()



# #%%

# # Start Figure Making
# fig = plt.figure(figsize=(30,15))

# # 4 Axes to make
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
# # Palaeolatitude against age
# ax1 = plt.subplot(gs[0])
# # Pass Percentage against age
# # ax2 = plt.subplot(gs[1],sharex=ax1)
# # # D_Median against age
# # ax3 = plt.subplot(gs[2],sharex=ax1)
# # # dC18O against age
# ax4 = plt.subplot(gs[1],sharex=ax1)

# # Plot the pass fail stripes:
# #     White = no data
# #     Lightgrey = data present, discarded by OPTiMAL
# #     Darkgrey = data present, used by OPTiMAL
    
# # Get the fail data
# df_choice = df_ancient_fail
# x = df_choice["Age"]

# df_distance = master_df_distance
# df_temp = master_df_ancient
# quartile = 0.5
# # Return the given slice of the D_Values
# df_distance = Return_Slice_of_Distance_df(df_distance,quartile,slice_option="Ancient")

# df_temp["D_values"] = df_distance

# # Age filter:
# # df_temp = df_temp[df_temp["Age"] >= 2]
# # df_temp = df_temp[df_temp["Age"] <= 4]

# unique_splitter = df_temp["Site"] 

# unique_values = set(unique_splitter)
# unique_values = list(unique_values)


# for key in unique_values:
    
#     # Get individual sites
#     # Take the log of the data
#     # Normalize the data
#     # Write out that data into a new column called D_values_norm
    
#     index = unique_values.index(key)
#     print(key)  
          
#     df_temp_site = df_temp[df_temp['Site'] == key]
#     df_temp_site = df_temp_site[df_temp_site["D_values"] < 14]
#     if len(df_temp_site.axes[0]) <= 5:
#         print(key) 
#         print(len(df_temp_site.axes[0]))
#     # elif key == "IODP U1463":
#     #     pass
#     else:
#         x_temp = df_temp_site["Age"].to_numpy()
#         y_temp = df_temp_site["D_values"].to_numpy()
        
#         x_max = np.max(x_temp)
#         x_min = np.min(x_temp)
#         x_range = x_max - x_min
#         res = 5/x_range
        
#         if res >= 1:
#             res = 1
        
#         # ax1.scatter(x_temp,y_temp,marker='X', s=100, alpha = 1)
    
#         smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=res)
    
    
#         #ax1.plot(x_temp,y_temp,c='r', alpha = 0.6, zorder= 2)
        
#         # With Legend:
#         # ax1.scatter(x_temp,y_temp,label=key, alpha = 0.2)   
#         # Without Legend: 
#         #ax1.scatter(x_temp,y_temp, alpha = 0.1)
#         # ax1.plot(smoothed[:, 0], smoothed[:, 1], 
#         #           c= 'k',
#         #           linewidth = 2,
#         #           alpha = 0.4,
#         #           zorder = 0)
        
#         pass_temp = df_temp_site[df_temp_site["D_Nearest"] <= 0.5]
#         fail_temp = df_temp_site[df_temp_site["D_Nearest"] > 0.5]
        
#         x_temp = fail_temp["Age"].to_numpy()
#         y_temp = fail_temp["D_values"].to_numpy()
#         ax1.scatter(x_temp,y_temp,marker = 'x', c = 'k', s=50,
#                     label="Fail", alpha = 0.5, zorder = 1) 
        
#         x_temp = pass_temp["Age"].to_numpy()
#         y_temp = pass_temp["D_values"].to_numpy()
#         ax1.scatter(x_temp,y_temp,marker = 'o', c = 'cornflowerblue',# edgecolors='black',
#                     linewidth=0.5 ,s = 150, label="Pass", alpha = 1, zorder = 1)  
            



# x_temp = df_temp["Age"].to_numpy()
# y_temp = df_temp["D_values"].to_numpy()
# x_max = np.max(x_temp)
# x_min = np.min(x_temp)
# x_range = x_max - x_min
# res = 2.5/x_range

# if res >= 1:
#     res = 1
# # ax1.scatter(x_temp,y_temp,marker='X', s=100, alpha = 1)

# smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=res)


# # # ax1.plot(x_temp,y_temp,label=key, alpha = 0.6, zorder= 2)
# # ax1.scatter(x_temp,y_temp, alpha = 0.3)

# ax1.plot(smoothed[:, 0], smoothed[:, 1], c= 'white',
#          linewidth=8, label="All", alpha = 1, zorder=4)

# ax1.plot(smoothed[:, 0], smoothed[:, 1], c= 'r',
#          linewidth=5, label="All", alpha = 0.6, zorder=5)

# # x_temp = df_ave["Age"]
# # y_temp = df_ave["D_values"]
  
# # res = 0.015

# # smoothed = sm.nonparametric.lowess(exog=x_temp, endog=y_temp, frac=res)

# # ax1.plot(smoothed[:, 0], smoothed[:, 1], c= 'white',
# #          linewidth=8, label="All", alpha = 1)

# # ax1.plot(smoothed[:, 0], smoothed[:, 1], c= 'teal',
# #          linewidth=5, label="All", alpha = 1)

# # Plots out a weighte

# ax1.set_xlim(0,193)
# # ax1.set_xlim(0,70)

# ax1.set_ylim(1,15)

# # Add dC18O Data
# x4 = df_loess["age_tuned"]
# y4 = df_loess["ISOBENd18oLOESSsmooth"]

# ax4.scatter(x4,y4,c='cornflowerblue',s=10,alpha=1, zorder=3)

# x4 = df_loess["age_tuned"]
# y4 = df_loess["ISOBENd18oLOESSsmoothLongTerm"]

# ax4.plot(x4,y4,'r', zorder=3)

# df_phan = df_phan[df_phan["Age"] <= 190]
# df_phan = df_phan[df_phan["Age"] >= 67]
# df_phan = df_phan[df_phan["ProxyType"] == "d18c"]

# x4 = df_phan["Age"]
# y4 = df_phan["ProxyValue"]

# # ax4 = ax3.twinx()
# ax4.scatter(x4,y4,c='cornflowerblue',s=10,alpha=1, zorder=2)

# df_out = df_phan
# df_Stat = Bin_and_Stat(df_out)

# x4 = df_Stat["pos"]
# y4 = df_Stat["mean"]
# ax4.plot(x4,y4,'r', zorder=3)

# # Adds a visual aid of Epochs along the bottom of the palaeolatitude plot
# x_box = 14
# x_thickness = 2
# # Pleistocene
# rect = patches.Rectangle((0,x_box),2.5,x_thickness,facecolor='wheat',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Pliocene
# rect = patches.Rectangle((2.5,x_box),2.8,x_thickness,facecolor='gold',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Miocene
# rect = patches.Rectangle((5.3,x_box),17.7,x_thickness,facecolor='yellow',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Oligocene
# rect = patches.Rectangle((23,x_box),10.9,x_thickness,facecolor='sandybrown',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Eocene
# rect = patches.Rectangle((33.9,x_box),22.1,x_thickness,facecolor='coral',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Palaeocene
# rect = patches.Rectangle((56.0,x_box),10,x_thickness,facecolor='tomato',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Cretaceous
# rect = patches.Rectangle((66.0,x_box),79,x_thickness,facecolor='limegreen',edgecolor='black',zorder=0)
# ax1.add_patch(rect)
# # Jurassic
# rect = patches.Rectangle((145,x_box),55,x_thickness,facecolor='cornflowerblue',edgecolor='black',zorder=0)
# ax1.add_patch(rect)

# # ax4.set_ylim((-7,7))
# ax4.set_ylim((-9,6))

# ax4.invert_yaxis()

# # ax1.set_xscale('log')
# # ax1.legend()

# plt.savefig("D_Values againts D18o.svg")       
     
        
     
        
     
        
     