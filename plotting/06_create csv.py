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
file_paths=['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS NEW/TBS007/REAL/TBS007_04 RDK shorter_2023-04-29_18h00.19.532.csv']
df_all= pd.DataFrame()

for file_path in file_paths:
    filename = os.path.basename(file_path)
    print(filename)
# Get the file extension from the filename
file_extension = os.path.splitext(filename)[1]
# Get the subject ID from the filename
subject_id = filename.split('_')[0]
field_position = filename.split('_')[1]

filenames_2 = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS NEW/TBS007/REAL/TBS007_04 RDK shorter_2023-04-29_18h29.37.292.csv']
df_all_2= pd.DataFrame()

# Loop through each CSV file and concatenate the data
for filename in file_paths:
    csv = pd.read_csv(filename)
    df = pd.DataFrame(csv, columns=['coherence', 'dir', 'key_resp.keys','key_resp.corr'])
    df = df.drop(df.index[-1])
    df_all = pd.concat([df_all, df])

# Loop through each CSV file and concatenate the data
for filename in filenames_2:
    csv = pd.read_csv(filename)
    df_2 = pd.DataFrame(csv, columns=['coherence', 'dir', 'key_resp.keys','key_resp.corr'])
    df_2 = df_2.drop(df_2.index[-1])
    df_all_2= pd.concat([df_all_2, df_2])


df_all['key_resp.keys'] = df_all['key_resp.keys'].apply(lambda x: 1 if x == 'p' else 0)
df_all_2['key_resp.keys'] = df_all_2['key_resp.keys'].apply(lambda x: 1 if x == 'p' else 0)

allintensity_0 = df_all.loc[df['dir'] == 0, 'coherence'].to_numpy()
allResponses_0 = df_all.loc[df['dir'] == 0, 'key_resp.keys'].to_numpy()
allintensity_180 = df_all.loc[df['dir'] == 180, 'coherence'].to_numpy()
allResponses_180 = df_all.loc[df['dir'] == 180, 'key_resp.keys'].to_numpy()

allintensity_0_2 = df_all_2.loc[df_2['dir'] == 0, 'coherence'].to_numpy()
allResponses_0_2 = df_all_2.loc[df_2['dir'] == 0, 'key_resp.keys'].to_numpy()
allintensity_180_2 = df_all_2.loc[df_2['dir'] == 180, 'coherence'].to_numpy()
allResponses_180_2 = df_all_2.loc[df_2['dir'] == 180, 'key_resp.keys'].to_numpy()

intensity_0, combinedResp_0, n = data.functionFromStaircase(
    allintensity_0, allResponses_0, 'unique')
intensity_180, combinedResp_180, n = data.functionFromStaircase(
    allintensity_180 * -1, allResponses_180, 'unique')

intensity_0_2, combinedResp_0_2, n = data.functionFromStaircase(
    allintensity_0_2, allResponses_0_2, 'unique')
intensity_180_2, combinedResp_180_2, n = data.functionFromStaircase(
    allintensity_180_2 * -1, allResponses_180_2, 'unique')


# Concatenate the two arrays into one
intensity = np.concatenate((intensity_0, intensity_180))
combinedResp = np.concatenate((combinedResp_0, combinedResp_180))
intensity_2 = np.concatenate((intensity_0_2, intensity_180_2))
combinedResp_2 = np.concatenate((combinedResp_0_2, combinedResp_180_2))


# Create a dictionary with the data
data_5 = {'subject_id':subject_id,'field_position':field_position, 'intensity': intensity, 'combinedResp': combinedResp,'intensity_2':intensity_2,'combinedResp_2':combinedResp_2}
# Create a DataFrame from the dictionary
df = pd.DataFrame(data_5)
# Export to Excel
df.to_excel(subject_id + '.xlsx', index=False)

# Concatenate the arrays
intensity = np.concatenate((intensity_0, intensity_180))
combinedResp = np.concatenate((combinedResp_0, combinedResp_180))
intensity_2 = np.concatenate((intensity_0_2, intensity_180_2))
combinedResp_2 = np.concatenate((combinedResp_0_2, combinedResp_180_2))


fit = data.FitLogistic(intensity, combinedResp, guess=[0.2,0.5],expectedMin=0)
smoothInt = np.arange(min(intensity), max(intensity), 0.01)
smoothResp= fit.eval(smoothInt)
pse = fit.inverse(0.5)
jnd = fit.inverse(0.75) - fit.inverse(0.5)

fit_2 = data.FitLogistic(intensity_2, combinedResp_2, guess=[0.2,0.5],expectedMin=0)
smoothInt_2 = np.arange(min(intensity_2), max(intensity_2), 0.01)
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


plt.xlim([-1, 1])
plt.text(1, -0.2, '(right)')
plt.text(-1, -0.2, '(left)')
plt.xlabel('Coherence level')
plt.ylabel('Proportion of choosing "moving to right"')
plt.title(subject_id + ' ' + field_position)

plt.legend()
plt.show()



