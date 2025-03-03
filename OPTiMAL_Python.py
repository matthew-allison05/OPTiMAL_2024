# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:24:19 2024

@author: Matthew

To Do List: 
    
    - Get OPTiMAL gaussian working in Python 
    - Demonstrate it makes the same predictions as in Matlab
        - Subset of the data - show SST, error and D_values are the same
    - If I can find a way to save the GP model, so you can just make predictions
      instead of running a model from scratch every single time.
    - Have OPTiMAL output formatted in the same way as MATLAB:
        - Excel sheets - calibration, ancient, sigmas, distance
    
    - List of functions to prepare: 
        - Global calibration maps
        - Return the D_value slices for chosen quartile
        - Return a D_values and SST prediction, showing data loss

"""

import os
import pandas as pd

# Get the code directory
code_dr = os.path.abspath(__file__)
name = os.path.basename(__file__)
name_length = len(name)
code_dr = code_dr[:-name_length]
# Just in case, change into the target directory
os.chdir(code_dr)

import OPTiMAL_Functions as fn

# Get the Matlab data

# Read files in - OPTiMAL_Output for Example_Holocene.xlsx
xls = pd.ExcelFile("OPTiMAL_Output_Holocene.xlsx")
df_calibration = pd.read_excel(xls, 'Calibration_Data_matlab')
df_ancient = pd.read_excel(xls, 'Ancient_Data_matlab')
df_distance = pd.read_excel(xls, 'Distance_Array_matlab')

xls.close()

# Read files in - OPTiMAL results for all ancient data (n = 12,256) - Can be a little 
xls = pd.ExcelFile("OPTiMAL_Output_All_Epochs.xlsx")
master_df_calibration = pd.read_excel(xls, 'Calibration_Data_matlab')
master_df_ancient = pd.read_excel(xls, 'Ancient_Data_matlab')
# This line can be slow to read in...
master_df_distance = pd.read_excel(xls, 'Distance_Array_matlab')

xls.close()


    # Example of how to return a slice of the Distance_Array from OPTiMAL.
    # In this case, return the median (quartile = 0.5) for the ancient dataset
    # and append that data to the ancient dataframe.
D_values = fn.Return_Slice_of_Distance_df(df_distance,0.5,slice_option="Ancient")
df_ancient["D_Values"] = D_values

    # Example of how to generate a calibration map. Just give the function the 
    # dataframe outputs from OPTiMAL. Some custom variables are customisable in
    # the function definition.
fn.Make_Calibration_Map(df_calibration, df_ancient, df_distance, save_fig=False)

    # Example of how to generate the SST timeseries from OPTiMAL against absolute
    # palaeolatitude. Can choose to see the QC failure rate of OPTiMAL's
    # D_Nearest <= 0.5.
    # Currently plots the entire 200 Ma dataset
fn.OPTiMAL_SST_Timeseries(master_df_calibration, master_df_ancient, save_fig=False)

    # Example of how to generate the SST timeseries from OPTiMAL but for a given 
    # Epoch's worth of data. Example given is the Palaeocene
df_ancient_Palaeocene = fn.Return_Given_Epoch_df(master_df_ancient,"Palaeocene")
fn.OPTiMAL_SST_Timeseries(df_calibration, df_ancient_Palaeocene, save_fig=False)

    # Example of how to generate the OPTiMAL D_Value timeseries plot.
    # Can choose to see the geological Epochs.
fn.OPTiMAL_DValue_Timeseries(df_calibration, master_df_ancient, master_df_distance, save_fig=False)

    # Example of how to generate the ODP 1168 and 1172 plot.
    # Includes TEX86 and OPTiMAL SST comparison with additional
    # D_Value plot.
fn.ODP_1168_1172(df_calibration, master_df_ancient, master_df_distance, save_fig=False)

    # Example of how to generate failure rate plots
    # Just specify an epoch
fn.Failure_Rates_Palaoelatitude('Eocene', master_df_ancient)
