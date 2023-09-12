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
filenames = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS discrimination/TBS07/TBS07_06 RDK fine discrimination_2023-05-12_20h46.40.714.csv']
# Create an empty DataFrame to store data from all CSV files
df_all = pd.DataFrame()

# Loop through each CSV file and concatenate the data
for filename in filenames:
    
    csv = pd.read_csv(filename)
    df = pd.DataFrame(csv, columns=['dir', 'key_resp.keys','key_resp.corr'])
    df = df.drop(df.index[-1])
    df_all = pd.concat([df_all, df])


#df_all['key_resp.keys'] = df_all['key_resp.keys'].apply(lambda x: 1 if x == 'p' else 0)

correct= df_all['key_resp.corr'].to_numpy()
dir=np.abs(df_all['dir'].to_numpy())

intensity, combinedResp, n = data.functionFromStaircase(
    dir, correct, 'unique')
fit = data.FitWeibull(intensity, combinedResp, guess=[0.2,0.5],expectedMin=0)
smoothInt = np.arange(min(intensity), max(intensity), 0.01)
smoothResp= fit.eval(smoothInt)
threshold=fit.inverse(0.75)
# Plot the data
plt.plot(intensity, combinedResp, 'bo')
plt.plot(smoothInt, smoothResp, '-')
plt.plot([threshold, threshold],[0,0.75],'--')
plt.plot([0, threshold],[0.75,0.75],'--')
plt.text(-0.5, 0.9, f'threshold: {threshold:.3f}',color='red')

plt.xlim([0, 50])
plt.ylim([0.5, 1])
plt.xlabel('direction')
plt.ylabel('correction rate')
plt.title('direction discrimination')
plt.legend()
plt.show()

