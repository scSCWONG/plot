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
filenames = ['/Users/sc/Downloads/DIANA_06 RDK fine discrimination_2023-05-08_18h01.11.368.csv']
# Create an empty DataFrame to store data from all CSV files
df_all = pd.DataFrame()

# Loop through each CSV file and concatenate the data
for filename in filenames:
    
    csv = pd.read_csv(filename)
    df = pd.DataFrame(csv, columns=['dir', 'key_resp.keys','key_resp.corr'])
    df = df.drop(df.index[-1])
    df_all = pd.concat([df_all, df])

# Arrange allintensity according to dot direction
#allintensity_0 = np.round(df_all.loc[df_all['dotDirection'] == 0, 'Coherence_staircase.intensity'].to_numpy(), 2)
#allintensity_180 = np.round(df_all.loc[df_all['dotDirection'] == 180, 'Coherence_staircase.intensity'].to_numpy(), 2)

# Assign negative value to allintensity_180
#df_all.loc[df['dir'] == 0, 'coherence'] = df_all.loc[df['dir'] == 0, 'coherence'].abs()
#df_all.loc[df['dir'] == 180, 'coherence'] = -df_all.loc[df['dir'] == 180, 'coherence'].abs()
df_all['key_resp.keys'] = df_all['key_resp.keys'].apply(lambda x: 1 if x == 'q' else 0)


allintensity_0 = df_all.loc[df['dir'] <0 , 'dir'].to_numpy()
allResponses_0 = df_all.loc[df['dir'] <0, 'key_resp.keys'].to_numpy()
allintensity_180 = df_all.loc[df['dir'] >0, 'dir'].to_numpy()
allResponses_180 = df_all.loc[df['dir'] >0, 'key_resp.keys'].to_numpy()

print(allintensity_0)
intensity_0, combinedResp_0, n = data.functionFromStaircase(
    allintensity_0, allResponses_0, 'unique')
intensity_180, combinedResp_180, n = data.functionFromStaircase(
    allintensity_180 , allResponses_180, 'unique')


# Concatenate the two arrays into one
intensity = np.concatenate((intensity_0, intensity_180))
combinedResp = np.concatenate((combinedResp_0, combinedResp_180))


fit = data.FitLogistic(intensity, combinedResp, guess=[0.2,0.5],expectedMin=0)
smoothInt = np.arange(min(intensity), max(intensity))
smoothResp= fit.eval(smoothInt)
pse = fit.inverse(0.5)
jnd = fit.inverse(0.75) - fit.inverse(0.5)
threshold=fit.inverse(0.75)
print(jnd)
# Plot the data
plt.plot(intensity, combinedResp, 'bo')
plt.plot(smoothInt, smoothResp, '-')
plt.plot([pse,pse],[0,0.5],'--');
plt.plot([-1, pse],[0.5,0.5],'--')
plt.xlim([-40,40])
plt.text(1, -0.2, '(right)')
plt.text(-1, -0.2, '(left)')
plt.xlabel('direction')
plt.text(0.5, 0.7, f'JND: {jnd:.3f}')
plt.ylabel('Proportion of choosing "moving to left"')
plt.title('RDK_1')
plt.legend()
plt.show()
