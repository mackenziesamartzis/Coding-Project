#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:26:27 2022
Modified on Fri Mar 11 15:26:27 2022
@author: Mackenzie Samartzis

Description
------------
"""

# Part 1 - Histogram the data

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(1,figsize=(6,6))
fig.clf()
axes = [fig.add_subplot(321),\
        fig.add_subplot(322),\
        fig.add_subplot(323),\
        fig.add_subplot(324),\
        fig.add_subplot(325),\
        fig.add_subplot(326)]

refractionData = np.loadtxt('refractionData.txt', skiprows = 3)

axes[0].hist(refractionData[0], bins = 15, range = (-10,50))
axes[0].set_xticks([-10,0,10,20,30,40,50])
axes[0].set_xlabel('beta (deg.)')
axes[0].set_title('alpha = 10 deg.')

axes[1].hist(refractionData[1], bins = 15, range = (-10,50))
axes[1].set_xticks([-10,0,10,20,30,40,50])
axes[1].set_xlabel('beta (deg.)')
axes[1].set_title('alpha = 20 deg.')

axes[2].hist(refractionData[2], bins = 15, range = (-10,50))
axes[2].set_xticks([-10,0,10,20,30,40,50])
axes[2].set_xlabel('beta (deg.)')
axes[2].set_title('alpha = 30 deg.')

axes[3].hist(refractionData[3], bins = 15, range = (-10,50))
axes[3].set_xticks([-10,0,10,20,30,40,50])
axes[3].set_xlabel('beta (deg.)')
axes[3].set_title('alpha = 40 deg.')

axes[4].hist(refractionData[4], bins = 15, range = (-10,50))
axes[4].set_xticks([-10,0,10,20,30,40,50])
axes[4].set_xlabel('beta (deg.)')
axes[4].set_title('alpha = 50 deg.')

axes[5].hist(refractionData[5], bins = 15, range = (-10,50))
axes[5].set_xticks([-10,0,10,20,30,40,50])
axes[5].set_xlabel('beta (deg.)')
axes[5].set_title('alpha = 60 deg.')

plt.tight_layout()

#%%

# Part 2 - Table of measurements

refractionData_radians = (refractionData)*np.pi/180

beta_mean10 = np.mean(refractionData_radians[0])
beta_std10 = np.std(refractionData_radians[0], ddof = 1)/np.sqrt(16)

beta_mean20 = np.mean(refractionData_radians[1])
beta_std20 = np.std(refractionData_radians[1], ddof = 1)/np.sqrt(16)

beta_mean30 = np.mean(refractionData_radians[2])
beta_std30 = np.std(refractionData_radians[2], ddof =1)/np.sqrt(16)

beta_mean40 = np.mean(refractionData_radians[3])
beta_std40 = np.std(refractionData_radians[3], ddof = 1)/np.sqrt(16)

beta_mean50 = np.mean(refractionData_radians[4])
beta_std50 = np.std(refractionData_radians[4], ddof = 1)/np.sqrt(16)

beta_mean60 = np.mean(refractionData_radians[5])
beta_std60 = np.std(refractionData_radians[5], ddof = 1)/np.sqrt(16)

beta_mean_set = np.array([beta_mean10, beta_mean20, beta_mean30, beta_mean40, beta_mean50, beta_mean60])

def mean_values(x):
    result = np.sin(x) 
    return result

mean_valueb10 = mean_values(beta_mean10)
mean_value10err = np.cos(beta_mean10)*beta_std10

mean_valueb20 = mean_values(beta_mean20)
mean_value20err = np.cos(beta_mean20)*beta_std20

mean_valueb30 = mean_values(beta_mean30)
mean_value30err = np.cos(beta_mean30)*beta_std30

mean_valueb40 = mean_values(beta_mean40)
mean_value40err = np.cos(beta_mean40)*beta_std40

mean_valueb50 = mean_values(beta_mean50)
mean_value50err = np.cos(beta_mean50)*beta_std50

mean_valueb60 = mean_values(beta_mean60)
mean_value60err = np.cos(beta_mean60)*beta_std60

sine_mean_set = np.array([mean_valueb10, mean_valueb20, mean_valueb30, mean_valueb40, mean_valueb50, mean_valueb60])
sine_sigma = np.array([mean_value10err, mean_value20err, mean_value30err, mean_value40err, mean_value50err, mean_value60err])

alpha_values = np.array([10,20,30,40,50,60])
alpha_values_radians = (alpha_values)*np.pi/180
sin_a10 = np.sin(alpha_values_radians[0])
sin_a20 = np.sin(alpha_values_radians[1])
sin_a30 = np.sin(alpha_values_radians[2])
sin_a40 = np.sin(alpha_values_radians[3])
sin_a50 = np.sin(alpha_values_radians[4])
sin_a60 = np.sin(alpha_values_radians[5])

sine_alphaset = np.array([sin_a10, sin_a20, sin_a30, sin_a40, sin_a50, sin_a60])

heading = '|   α   | sin(α) | the mean value of β | the sine of the mean value of β | σ(sin(β))  | '
line = '+'*86
print(line) 
print(heading)
print(line)

for i in range(0,6):
    print(f'| {alpha_values_radians[i]:.3f} | {sine_alphaset[i]:.3f}  |       {beta_mean_set[i]:.3f}         |              {sine_mean_set[i]:.3f}              |    {sine_sigma[i]:.3f}   |')
    print(line)


#%%

# Part 3 - Snells law plot and fit

import scipy.stats as st
from scipy.optimize import curve_fit


fig = plt.figure(2,figsize=(6,6))
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.errorbar(sine_alphaset, sine_mean_set, sine_sigma, fmt = 'ok', label = 'Original Data')
ax1.set_xlabel('sin(α)', fontsize = 14)
ax1.set_ylabel('sin(β)', fontsize = 14)




def snells_law(sina, n):
    sinb = sina/n
    return sinb


p0 = (1,)

params, covar = curve_fit(snells_law, sine_alphaset, sine_mean_set, p0, sigma = sine_sigma, absolute_sigma = True)
nfit = params[0]
nerr = np.sqrt(covar[0][0])
print(f'The best fit value of the index of refraction is {nfit:.3f} +/- {nerr:.3f}')

ax1.plot(sine_alphaset, snells_law(sine_alphaset, nfit), '-', label = "Line of Best Fit")
ax1.set_title("Snell's Law", fontsize = 16)
ax1.legend()
plt.tight_layout()

npt = 5 #number of points is degrees of freedom

chisq = np.sum((sine_mean_set-snells_law(sine_alphaset, nfit))**2/(sine_sigma)**2)

#chisq = np.sum((sine_mean_set-snells_law(sine_alphaset, nfit))**2/(sine_sigma)**2)
pvalue = st.chi2.sf(chisq, npt)
print(f'The chi-squared value for the best fit line compared to the data is {chisq:.2f}')
print(f'The p-value for the best fit line compared to the data is {pvalue:.2f}')
print(f'The number of degrees of freedom is {npt}')

# Part 4 - Chi squared plot
nvalues = np.linspace(1,2,1000)
chisq2 = np.zeros(len(nvalues))

for i in range(len(chisq2)):
    chisq2[i] = np.sum((sine_mean_set - snells_law(sine_alphaset, nvalues[i]))**2/sine_sigma**2)

ax2.plot(nvalues, chisq2)       
ax2.set_xlabel('Index of Refraction', fontsize = 14)
ax2.set_ylabel('$\\chi^2$', fontsize = 14)     


chi2min = chisq2.min()
ax2.set_ylim(2.5,5)
ax2.set_xlim(1.45, 1.53)
ax2.hlines(chi2min,1.45,1.53, color = 'black')
ax2.hlines(chi2min+1.0,1.45,1.53, color = 'black')
ax2.vlines(nfit+nerr,0.0,chi2min+1,ls='--', color = 'black')
ax2.vlines(nfit-nerr,0.0,chi2min+1,ls='--', color = 'black')
ax2.vlines(nfit, 0.0, chi2min+1, color = 'black')

ax2.text(1.475, 4.7, f' n = {nfit:.3f} $\\pm$ {nerr:.3f}', fontsize = 14)
ax2.set_title('$\\chi^2$ vs slope', fontsize = 16)

plt.tight_layout()
      




######################################################
