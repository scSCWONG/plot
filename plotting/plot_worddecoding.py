import os
from datetime import datetime
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sy
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import pylab
from psychopy import data, gui, core
from psychopy.tools.filetools import fromFile


# Define the filenames of the CSV files to read
filenames = ['/Users/sc/Desktop/Psychopy/w.csv']
# Create an empty DataFrame to store data from all CSV files
df_all = pd.DataFrame()

# Loop through each CSV file and concatenate the data
for filename in filenames:
    
    csv = pd.read_csv(filename)
    df = pd.DataFrame(csv, columns=['current_duration', 'image_type', 'rt','correction'])
    df_all = pd.concat([df_all, df])

# Arrange allintensity according to dot direction
#allintensity_0 = np.round(df_all.loc[df_all['dotDirection'] == 0, 'Coherence_staircase.intensity'].to_numpy(), 2)
#allintensity_180 = np.round(df_all.loc[df_all['dotDirection'] == 180, 'Coherence_staircase.intensity'].to_numpy(), 2)

# Assign negative value to allintensity_180
#df_all.loc[df['dir'] == 0, 'coherence'] = df_all.loc[df['dir'] == 0, 'coherence'].abs()
#df_all.loc[df['dir'] == 180, 'coherence'] = -df_all.loc[df['dir'] == 180, 'coherence'].abs()
#df_all['key_resp.keys'] = df_all['key_resp.keys'].apply(lambda x: 1 if x == 'p' else 0)

#p = np.concatenate([p0, p_180,q_0,q_180])
#print( df_all['coherence'])

allintensity_real = df_all.loc[df['image_type'] == 'real', 'current_duration'].to_numpy()
allResponses_real = df_all.loc[df['image_type'] == 'real', 'correction'].to_numpy()
allintensity_pseudo = df_all.loc[df['image_type'] == 'pseudo', 'current_duration'].to_numpy()
allResponses_pseudo = df_all.loc[df['image_type'] == 'pseudo', 'correction'].to_numpy()


intensity, combinedResp, n = data.functionFromStaircase(
    allintensity_real, allResponses_real, 'unique')
intensity_pseudo, combinedResp_pseudo, n = data.functionFromStaircase(
    allintensity_pseudo, allResponses_pseudo, 'unique')


fit = data.FitLogistic(intensity, combinedResp, guess=[0.02,0.5],expectedMin=0.5)
smoothInt = np.arange(min(intensity), max(intensity), 0.01)
smoothResp= fit.eval(smoothInt)
threshold=fit.inverse(0.75)


fit_2 = data.FitLogistic(intensity_pseudo, combinedResp_pseudo, guess=[0.02, 0.5], expectedMin=0.5)
smoothInt_2 = np.arange(min(intensity_pseudo), max(intensity_pseudo), 0.01)
smoothResp_2 = fit.eval(smoothInt_2)
threshold_2 = fit_2.inverse(0.75)

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot for real trials
axes[0].plot(intensity, combinedResp, 'bo')
axes[0].plot(smoothInt, smoothResp, '-')
axes[0].set_xlim([min(intensity), max(intensity)])
axes[0].set_xlabel('word duration')
axes[0].set_ylabel('Correction rate')
axes[0].set_title('Real Trials')

# Plot for pseudo trials
axes[1].plot(intensity_pseudo, combinedResp_pseudo, 'bo')
axes[1].plot(smoothInt_2, smoothResp_2, '-')
axes[1].set_xlim([min(intensity_pseudo), max(intensity_pseudo)])
axes[1].set_xlabel('word duration')
axes[1].set_ylabel('Correction rate')
axes[1].set_title('Pseudo Trials')

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()

