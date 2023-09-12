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

filenames = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/plot/TBS001.xlsx','/Users/sc/Desktop/Psychopy/Psychopy staircase/plot/TBS003.xlsx','/Users/sc/Desktop/Psychopy/Psychopy staircase/plot/TBS004.xlsx','/Users/sc/Desktop/Psychopy/Psychopy staircase/plot/TBS005.xlsx']
df_all= pd.DataFrame()


for filename in filenames:
    df = pd.read_excel(filename, usecols=['subject_id', 'field_position', 'intensity', 'combinedResp', 'intensity_2', 'combinedResp_2'])
    df = df.dropna()
    df_all = pd.concat([df_all, df])

# Group by subject ID and combinedResp for pre and post
df_mean = df_all.groupby(['intensity'])['combinedResp'].mean()
df_std = df_all.groupby(['intensity'])['combinedResp'].std()
df_mean_post = df_all.groupby(['intensity_2'])['combinedResp_2'].mean()
df_std_post = df_all.groupby(['intensity_2'])['combinedResp_2'].std()


mean_pre = df_mean.to_numpy()
std_pre = df_std.to_numpy()
mean_post = df_mean_post.to_numpy()
std_post = df_std_post.to_numpy()

intensity = list(np.arange(-0.6, 0.7, 0.1))
intensity_2= list(np.arange(-0.6, 0.7, 0.1))
len(mean_pre)
print(mean_post)
fit = data.FitLogistic(intensity, mean_pre, guess=[0.2,0.5],expectedMin=0)
smoothInt = np.arange(min(intensity), max(intensity), 0.01)
smoothResp= fit.eval(smoothInt)
pse = fit.inverse(0.5)
jnd = fit.inverse(0.75) - fit.inverse(0.5)

fit_2 = data.FitLogistic(intensity_2, mean_post, guess=[0.2,0.5],expectedMin=0)
smoothInt_2 = np.arange(min(intensity_2), max(intensity_2), 0.01)
smoothResp_2= fit_2.eval(smoothInt_2)
pse_2 = fit_2.inverse(0.5)
jnd_2 = fit_2.inverse(0.75) - fit_2.inverse(0.5)

# Plot the data
plt.plot(intensity, mean_pre, 'bo',label='Pre')
plt.errorbar(intensity, mean_pre, yerr=std_pre, fmt='bo')
plt.plot(smoothInt, smoothResp, '-', color='blue')
plt.plot([pse,pse],[0,0.5],'--',color='blue')
plt.plot([-1, pse],[0.5,0.5],'--',color='blue')
plt.text(0.5, 0.8, f'JND: {jnd:.3f}',color='blue')

plt.plot(intensity_2, mean_post, 'ro', label='Post')
plt.plot(smoothInt_2, smoothResp_2, '-',color='red')
plt.errorbar(intensity, mean_post, yerr=std_post, fmt='ro')
plt.plot([pse_2,pse_2],[0,0.5],'--',color='red')
plt.plot([-1, pse_2],[0.5,0.5],'--',color='red')
plt.text(0.5, 0.7, f'JND: {jnd_2:.3f}',color='red')
plt.xlim(-0.6, 0.6) # Set the x-limit to (-0.6, 0.6)
plt.xlabel('Coherence level')
plt.ylabel('Proportion of choosing "moving to right"')
plt.title('Real stimulation_right')
plt.legend()

plt.show()
