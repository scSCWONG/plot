import os
from datetime import datetime
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sy
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import pylab
from psychopy import data, gui, core
from psychopy.tools.filetools import fromFile


# Define the filenames of the CSV files to read
filenames = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS NEW/TBS001/VERTEX/TBS001_04 RDK shorter_2023-05-05_12h21.56.359.csv']
df_all = pd.DataFrame()
filenames_2 = ['/Users/sc/Desktop/Psychopy/Psychopy staircase/data figure/TBS NEW/TBS001/VERTEX/TBS001_04 RDK shorter_2023-05-05_12h48.37.695.csv']
df_all_2= pd.DataFrame()

# Loop through each CSV file and concatenate the data
for filename in filenames:
    csv = pd.read_csv(filename)
    df = pd.DataFrame(csv, columns=['coherence', 'dir', 'key_resp.keys','key_resp.corr'])
    df = df.drop(df.index[-1])
    df_all = pd.concat([df_all, df])
for filename in filenames_2:
    csv = pd.read_csv(filename)
    df_2 = pd.DataFrame(csv, columns=['coherence', 'dir', 'key_resp.keys','key_resp.corr'])
    df_2 = df_2.drop(df_2.index[-1])
    df_all_2= pd.concat([df_all_2, df_2])


# Categorize data according to dotDirection
allintensity_0 = df_all.loc[df['dir'] == 0, 'coherence'].to_numpy()
allResponses_0 = df_all.loc[df['dir'] == 0, 'key_resp.corr'].to_numpy()
allintensity_180 = df_all.loc[df['dir'] == 180, 'coherence'].to_numpy()
allResponses_180 = df_all.loc[df['dir'] == 180, 'key_resp.corr'].to_numpy()

allintensity_0_2 = df_all_2.loc[df_2['dir'] == 0, 'coherence'].to_numpy()
allResponses_0_2 = df_all_2.loc[df_2['dir'] == 0, 'key_resp.corr'].to_numpy()
allintensity_180_2 = df_all_2.loc[df_2['dir'] == 180, 'coherence'].to_numpy()
allResponses_180_2 = df_all_2.loc[df_2['dir'] == 180, 'key_resp.corr'].to_numpy()


intensity_0, combinedResp_0, n = data.functionFromStaircase(
    allintensity_0, allResponses_0, 'unique')
intensity_180, combinedResp_180, n = data.functionFromStaircase(
    allintensity_180, allResponses_180, 'unique')


intensity_0_2, combinedResp_0_2, n = data.functionFromStaircase(
    allintensity_0_2, allResponses_0_2, 'unique')
intensity_180_2, combinedResp_180_2, n = data.functionFromStaircase(
    allintensity_180_2, allResponses_180_2, 'unique')


# Fit curve - in this case using a Weibull function
fit = data.FitWeibull(intensity_0, combinedResp_0, guess=[0.2, 0.5])
smoothInt_0 = pylab.arange(min(intensity_0), max(intensity_0), 0.01)
smoothResp_0 = fit.eval(smoothInt_0)
thresh_0 = fit.inverse(0.75)
print(thresh_0)

fit_180 = data.FitWeibull(intensity_180, combinedResp_180, guess=[0.2, 0.5])
smoothInt_180 = pylab.arange(min(intensity_180), max(intensity_180), 0.01)
smoothResp_180 = fit_180.eval(smoothInt_180)
thresh_180 = fit_180.inverse(0.75)
print(thresh_180)

fit_2 = data.FitWeibull(intensity_0_2, combinedResp_0_2, guess=[0.2,0.5])
smoothInt_0_2 = pylab.arange(min(intensity_0_2), max(intensity_0_2), 0.01)
smoothResp_0_2= fit_2.eval(smoothInt_0_2)
thresh_0_post = fit_2.inverse(0.75)
print(thresh_0_post)

fit_2_180 = data.FitWeibull(intensity_180_2, combinedResp_180_2, guess=[0.2,0.5])
smoothInt_180_2 = pylab.arange(min(intensity_180_2), max(intensity_180_2), 0.01)
smoothResp_180_2= fit_2_180.eval(smoothInt_180_2)
thresh_180_post = fit_2_180.inverse(0.75)
print(thresh_180_post)

pylab.subplot(121)
pylab.plot(smoothInt_0, smoothResp_0, '-')
pylab.plot([thresh_0, thresh_0],[0,0.75],'--');
pylab.plot([0, thresh_0],[0.75,0.75],'--')
pylab.title('threshold_0 = %0.3f' %(thresh_0))
pylab.plot(intensity_0, combinedResp_0, 'o')
pylab.ylim([0.5,1])
pylab.xlim([0,1])

pylab.subplot(122)
pylab.plot(smoothInt_180, smoothResp_180, '-')
pylab.plot([thresh_180, thresh_180],[0,0.75],'--');
pylab.plot([0, thresh_180],[0.75,0.75],'--')
pylab.title('threshold_180 = %0.3f' %(thresh_180))
pylab.plot(intensity_180, combinedResp_180, 's')  # 's' stands for square marker
pylab.ylim([0.5,1])
pylab.xlim([0,1])

pylab.show()


pylab.subplot(221)
pylab.plot(smoothInt_0_2, smoothResp_0_2, '-')
pylab.plot([thresh_0_post, thresh_0_post],[0,0.75],'--');
pylab.plot([0, thresh_0_post],[0.75,0.75],'--')
pylab.title('Post_threshold_0 = %0.3f' %(thresh_0_post))
pylab.plot(intensity_0_2, combinedResp_0_2, 'o')
pylab.ylim([0.5,1])
pylab.xlim([0,1])

pylab.subplot(222)
pylab.plot(smoothInt_180_2, smoothResp_180_2, '-')
pylab.plot([thresh_180_post, thresh_180_post],[0,0.75],'--');
pylab.plot([0,thresh_180_post],[0.75,0.75],'--')
pylab.title('Post_threshold_180 = %0.3f' %(thresh_180_post))
pylab.plot(intensity_180_2, combinedResp_180_2, 's')  # 's' stands for square marker
pylab.ylim([0.5,1])
pylab.xlim([0,1])


pylab.show()


