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
filenames = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS discrimination/TBS07/v5/TBS07_06 RDK fine discrimination_2023-05-13_18h59.44.879.csv']
df_all= pd.DataFrame()

filenames_2 = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS discrimination/TBS07/v5/TBS07_06 RDK fine discrimination_2023-05-13_19h23.35.536.csv']
df_all_2= pd.DataFrame()

# Loop through each CSV file and concatenate the data
for filename in filenames:
    csv = pd.read_csv(filename)
    df = pd.DataFrame(csv, columns=['dir', 'key_resp.keys','key_resp.corr'])
    df = df.drop(df.index[-1])
    df_all = pd.concat([df_all, df])

# Loop through each CSV file and concatenate the data
for filename in filenames_2:
    csv = pd.read_csv(filename)
    df_2 = pd.DataFrame(csv, columns=['dir', 'key_resp.keys','key_resp.corr'])
    df_2 = df_2.drop(df_2.index[-1])
    df_all_2= pd.concat([df_all_2, df_2])

df_all['key_resp.keys'] = df_all['key_resp.keys'].apply(lambda x: 1 if x == 'q' else 0)
df_all_2['key_resp.keys'] = df_all_2['key_resp.keys'].apply(lambda x: 1 if x == 'q' else 0)

allintensity_0 = df_all.loc[df['dir'] <0 , 'dir'].to_numpy()
allResponses_0 = df_all.loc[df['dir'] <0, 'key_resp.keys'].to_numpy()
allintensity_180 = df_all.loc[df['dir'] >0, 'dir'].to_numpy()
allResponses_180 = df_all.loc[df['dir'] >0, 'key_resp.keys'].to_numpy()

allintensity_0_2 = df_all_2.loc[df_2['dir']<0 , 'dir'].to_numpy()
allResponses_0_2 = df_all_2.loc[df_2['dir'] <0, 'key_resp.keys'].to_numpy()
allintensity_180_2 = df_all_2.loc[df_2['dir'] >0, 'dir'].to_numpy()
allResponses_180_2 = df_all_2.loc[df_2['dir'] >0, 'key_resp.keys'].to_numpy()

intensity_0, combinedResp_0, n = data.functionFromStaircase(
    allintensity_0, allResponses_0, 'unique')
intensity_180, combinedResp_180, n = data.functionFromStaircase(
    allintensity_180 , allResponses_180, 'unique')

intensity_0_2, combinedResp_0_2, n = data.functionFromStaircase(
    allintensity_0_2, allResponses_0_2, 'unique')
intensity_180_2, combinedResp_180_2, n = data.functionFromStaircase(
    allintensity_180_2 , allResponses_180_2, 'unique')


# Concatenate the two arrays into one
intensity = np.concatenate((intensity_0, intensity_180))
combinedResp = np.concatenate((combinedResp_0, combinedResp_180))
intensity_2 = np.concatenate((intensity_0_2, intensity_180_2))
combinedResp_2 = np.concatenate((combinedResp_0_2, combinedResp_180_2))


fit = data.FitLogistic(intensity, combinedResp, guess=[0.2,0.5],expectedMin=0)
smoothInt = np.arange(min(intensity), max(intensity))
smoothResp= fit.eval(smoothInt)
pse = fit.inverse(0.5)
jnd = fit.inverse(0.75) - fit.inverse(0.5)


fit_2 = data.FitLogistic(intensity_2, combinedResp_2, guess=[0.2,0.5],expectedMin=0)
smoothInt_2 = np.arange(min(intensity_2), max(intensity_2))
smoothResp_2= fit_2.eval(smoothInt_2)
pse_2 = fit_2.inverse(0.5)
jnd_2 = fit_2.inverse(0.75) - fit_2.inverse(0.5)


# Plot the data
plt.plot(intensity, combinedResp, 'bo',label='Pre')
plt.plot(smoothInt, smoothResp, '-', color='blue')
plt.plot([pse,pse],[0,0.5],'--',color='blue')
plt.plot([-1, pse],[0.5,0.5],'--',color='blue')
plt.text(0.5, 0.8, f'JND: {jnd:.3f}',color='blue')

plt.plot(intensity_2, combinedResp_2, 'ro', label='Post')
plt.plot(smoothInt_2, smoothResp_2, '-',color='red')
plt.plot([pse_2,pse_2],[0,0.5],'--',color='red')
plt.plot([-1, pse_2],[0.5,0.5],'--',color='red')
plt.text(0.5, 0.7, f'JND: {jnd_2:.3f}',color='red')


plt.xlim([-40, 40])

plt.xlabel('Coherence level')
plt.ylabel('Proportion of choosing "moving to right"')

# Initialize variables
tbs_part = ''
v5_list = []

for filename in filenames:
    # Extract "TBS01" from the filename
    tbs_part = os.path.basename(os.path.dirname(filename))
    
    # Extract "V5" from the filename
    v5_part = os.path.basename(os.path.dirname(os.path.dirname(filename)))
    v5_list.append(v5_part)
    
    print(filename)


# Combine "TBS01" and "V5" for the title
title = f"{v5_list[0]} {tbs_part}"
# Set the combined title in the plot
plt.title(title)

# Save the figure
save_dir = os.path.dirname(filenames[0])
save_filename = title  # Specify the desired filename with extension
save_path = os.path.join(save_dir, save_filename)
plt.savefig(save_path)
plt.legend()

plt.show()


