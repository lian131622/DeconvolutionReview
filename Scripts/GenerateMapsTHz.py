import os
import Helper
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Import Reference
path = './Data/NewTHz.mat'
Reference = loadmat(path)['ref'].squeeze()
Reference = Helper.Normalize(Reference)

Subsample = 1
Reference = Reference[::Subsample]
FWHM = Helper.FWHM(Reference)

#%% Import data and create Maps
path = 'Maps/DataTHz'
fileMAT = os.listdir(path)

order = np.array([8,11,5,2,7,10,4,1,6,9,3,0])

fileMAT = [fileMAT[i] for i in order]

ErrorAlpha1 = []
ErrorAlpha2 = []
ErrorTau1 = []
ErrorTau2 = []
FOMDelay = []
FOMRatio = []
for file in fileMAT:
    pathMAT = 'Maps/DataTHz/'+str(file)
    Data = loadmat(pathMAT)
    alpha = Data['alpha'].squeeze() # Vector of amplitude of the second peak
    lam = Data['lam'].squeeze() # Vector of delay in FWHM
    SNR_dB = Data['SNR_dB'].squeeze()
    TauS = Data['TauS'] # Matrix Nx2: [Position of the first peak, Position of the second peak]
    AlphaS = Data['AlphaS'] # Matrix Nx2: [Amplitude of the first peak, Amplitude of the second peak]
    TauMaps = Data['TauMaps'] # Matrix 4D: [NMean, N, Algorithm ,Position reconstructed, Second poistion peak]
    AlphaMaps = Data['AlphaMaps'] # Same as TauMaps but for the amplitude.
    
    # Compute the mean
    TauMapsMean = np.mean(TauMaps,axis=0)
    AlphaMapsMean = np.mean(AlphaMaps,axis=0)
    
    # FOM ToF
    FOM_FirstToF = (np.abs(TauMapsMean[:,0]-TauS[:,0])/FWHM).reshape(len(alpha),len(lam))
    FOM_SecondToF = (np.abs(TauMapsMean[:,1]-TauS[:,1])/FWHM).reshape(len(alpha),len(lam))
     
    DelayS = np.abs(TauS[:,1]-TauS[:,0])
    DelayR = np.abs(TauMapsMean[:,1]-TauMapsMean[:,0])
    FOM_Delay = (np.abs(DelayS-DelayR)/DelayS).reshape(len(alpha),len(lam))
     
    # FOM Amplitude
    FOM_FirstPeak = (np.abs(AlphaMapsMean[:,0]-AlphaS[:,0])/np.abs(AlphaS[:,0])).reshape(len(alpha),len(lam))
    FOM_SecondPeak = (np.abs(AlphaMapsMean[:,1]-AlphaS[:,1])/np.abs(AlphaS[:,1])).reshape(len(alpha),len(lam))
     
    RatioS = AlphaS[:,1]/AlphaS[:,0]
    RatioR = AlphaMapsMean[:,1]/AlphaMapsMean[:,0]
    FOM_Ratio = (np.abs(RatioS-RatioR)/np.abs(RatioS)).reshape(len(alpha),len(lam))
    
    ErrorAlpha1.append(FOM_FirstPeak)
    ErrorAlpha2.append(FOM_SecondPeak )
    ErrorTau1.append(FOM_FirstToF)
    ErrorTau2.append(FOM_SecondToF)
    FOMDelay.append(FOM_Delay)
    FOMRatio.append(FOM_Ratio)
#%%
import matplotlib

plt.close('all')

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Maps Tau
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True,constrained_layout=True)
fig.set_size_inches(12, 7)

fig.suptitle(r'THz Data: $FOM_\tau = \frac{|(\hat{\tau_2}-\hat{\tau_1}) - (\tau_2-\tau_1)|}{|(\tau_2-\tau_1)|}$',fontsize=18)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(FOMDelay[i],aspect='auto',cmap='gray_r',vmin=0,vmax=1)
    ax.set_yticks([0,8,16,23,31,39],labels = [1,0.6,0.2,-0.2,-0.6,-1],fontsize=10)
    ax.set_xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=10,rotation=90)

    if i>7:
        ax.set_xlabel(r'$\tau$ [FWHM]',fontsize=15)
    if i%4==0:
        ax.set_ylabel(r'$\alpha$',fontsize=15)

for ax, col in zip(axes[0,:], ['OMP', 'SD', 'MUSIC','AR']):
    ax.annotate(col, (0.5, 1), xytext=(0, 10), ha='center', va='bottom',
                size=18, xycoords='axes fraction', textcoords='offset points')

for ax, row in zip(axes[:,0], [r'$FOM_{\tau}$'+'\n@20dB', r'$FOM_{\tau}$'+'\n@10dB', r'$FOM_{\tau}$'+'\n@0dB']):
    ax.annotate(row, (0, 0.5), xytext=(-70, 0), ha='center', va='center',
                size=18, xycoords='axes fraction',
                textcoords='offset points')
# Create a colorbar
cbar = fig.colorbar(im,ax=axes.ravel(),pad=0.02,aspect=15)

plt.savefig('Maps/Maps_THz_Tau.png',dpi=600)

# Maps Alpha

fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True,constrained_layout=True)
fig.set_size_inches(12, 7)

fig.suptitle(r'THz Data: $FOM_\alpha = \frac{|\frac{\hat{\alpha_2}}{\hat{\alpha_1}} -\frac{\alpha_2}{\alpha_1}|}{|\frac{\alpha_2}{\alpha_1}|}$',fontsize=18)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(FOMRatio[i],aspect='auto',cmap='gray_r',vmin=0,vmax=1)
    ax.set_yticks([0,8,16,23,31,39],labels = [1,0.6,0.2,-0.2,-0.6,-1],fontsize=10)
    ax.set_xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=10,rotation=90)

    if i>7:
        ax.set_xlabel(r'$\tau$ [FWHM]',fontsize=15)
    if i%4==0:
        ax.set_ylabel(r'$\alpha$',fontsize=15)

for ax, col in zip(axes[0,:], ['OMP', 'SD', 'MUSIC','AR']):
    ax.annotate(col, (0.5, 1), xytext=(0, 10), ha='center', va='bottom',
                size=18, xycoords='axes fraction', textcoords='offset points')

for ax, row in zip(axes[:,0], [r'$FOM_{\alpha}$'+'\n@20dB', r'$FOM_{\alpha}$'+'\n@10dB', r'$FOM_{\alpha}$'+'\n@0dB']):
    ax.annotate(row, (0, 0.5), xytext=(-70, 0), ha='center', va='center',
                size=18, xycoords='axes fraction',
                textcoords='offset points')
# Create a colorbar
cbar = fig.colorbar(im,ax=axes.ravel(),pad=0.02,aspect=15)

plt.savefig('Maps/Maps_THz_Alpha.png',dpi=600)

#%%
# Maps Error Tau1
fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True,constrained_layout=True)
fig.set_size_inches(12, 7)

fig.suptitle(r'THz Data: Error ${\tau_1} = \frac{|(\hat{\tau_1}-{\tau_1})|}{FWHM}$',fontsize=18)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(ErrorTau1[i],aspect='auto',cmap='gray_r',vmin=0,vmax=1)
    ax.set_yticks([0,8,16,23,31,39],labels = [1,0.6,0.2,-0.2,-0.6,-1],fontsize=10)
    ax.set_xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=10,rotation=90)

    if i>7:
        ax.set_xlabel(r'$\tau$ [FWHM]',fontsize=15)
    if i%4==0:
        ax.set_ylabel(r'$\alpha$',fontsize=15)

for ax, col in zip(axes[0,:], ['OMP', 'SD', 'MUSIC','AR']):
    ax.annotate(col, (0.5, 1), xytext=(0, 10), ha='center', va='bottom',
                size=18, xycoords='axes fraction', textcoords='offset points')

for ax, row in zip(axes[:,0], [r'Error ${\tau_1}$'+'\n@20dB', r'Error ${\tau_1}$'+'\n@10dB', r'Error ${\tau_1}$'+'\n@0dB']):
    ax.annotate(row, (0, 0.5), xytext=(-70, 0), ha='center', va='center',
                size=18, xycoords='axes fraction',
                textcoords='offset points')
# Create a colorbar
cbar = fig.colorbar(im,ax=axes.ravel(),pad=0.02,aspect=15)

plt.savefig('Maps/Maps_THz_ErrorTau1.png',dpi=600)

# Maps Error Tau2

fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True,constrained_layout=True)
fig.set_size_inches(12, 7)

fig.suptitle(r'THz Data: Error ${\tau_2} = \frac{|(\hat{\tau_2}-{\tau_2})|}{FWHM}$',fontsize=18)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(ErrorTau2[i],aspect='auto',cmap='gray_r',vmin=0,vmax=1)
    ax.set_yticks([0,8,16,23,31,39],labels = [1,0.6,0.2,-0.2,-0.6,-1],fontsize=10)
    ax.set_xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=10,rotation=90)

    if i>7:
        ax.set_xlabel(r'$\tau$ [FWHM]',fontsize=15)
    if i%4==0:
        ax.set_ylabel(r'$\alpha$',fontsize=15)

for ax, col in zip(axes[0,:], ['OMP', 'SD', 'MUSIC','AR']):
    ax.annotate(col, (0.5, 1), xytext=(0, 10), ha='center', va='bottom',
                size=18, xycoords='axes fraction', textcoords='offset points')

for ax, row in zip(axes[:,0], [r'Error ${\tau_2}$'+'\n@20dB', r'Error ${\tau_2}$'+'\n@10dB', r'Error ${\tau_2}$'+'\n@0dB']):
    ax.annotate(row, (0, 0.5), xytext=(-70, 0), ha='center', va='center',
                size=18, xycoords='axes fraction',
                textcoords='offset points')
# Create a colorbar
cbar = fig.colorbar(im,ax=axes.ravel(),pad=0.02,aspect=15)

plt.savefig('Maps/Maps_THz_ErrorTau2.png',dpi=600)

#%%
import matplotlib.patches as mpatches

def CreateBestMaps(FOM,ColorLabel):
    ColorMap = np.zeros([len(alpha),len(lam),3])
    
    ColorMap[FOM==0] = ColorLabel[0]
    ColorMap[FOM==1] = ColorLabel[1]
    ColorMap[FOM==2] = ColorLabel[2]
    ColorMap[FOM==3] = ColorLabel[3]
    
    return ColorMap

# ColorMaps DragonBall
ColorLabel = np.array([[231,106,36],[28,69,149],[231,229,232],[1,8,10]])/255 # Orange,Gray,Blue,Black

FOMBest20Tau = CreateBestMaps(np.argmin(np.array(FOMDelay)[0:4],axis=0),ColorLabel)
FOMBest10Tau = CreateBestMaps(np.argmin(np.array(FOMDelay)[4:8],axis=0),ColorLabel)
FOMBest0Tau = CreateBestMaps(np.argmin(np.array(FOMDelay)[8:12],axis=0),ColorLabel)
FOMBestTau = np.array([FOMBest20Tau,FOMBest10Tau,FOMBest0Tau])

FOMBest20Alpha = CreateBestMaps(np.argmin(np.array(FOMRatio)[0:4],axis=0),ColorLabel)
FOMBest10Alpha = CreateBestMaps(np.argmin(np.array(FOMRatio)[4:8],axis=0),ColorLabel)
FOMBest0Alpha = CreateBestMaps(np.argmin(np.array(FOMRatio)[8:12],axis=0),ColorLabel)
FOMBestAlpha = np.array([FOMBest20Alpha,FOMBest10Alpha,FOMBest0Alpha])

OMPPatch = mpatches.Patch(color=ColorLabel[0], label='OMP')
SDPatch = mpatches.Patch(color=ColorLabel[1], label='SD')
MUSICPatch = mpatches.Patch(color=ColorLabel[2], label='MUSIC')
ARPatch = mpatches.Patch(color=ColorLabel[3], label='AR')

#%%
# Maps Best Tau
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True,constrained_layout=True)
fig.set_size_inches(12, 4)

fig.suptitle(r'THz Data: $FOM_{\tau}$ Best algorithm vs SNR',fontsize=18)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(FOMBestTau[i],aspect='auto',cmap='gray_r',vmin=0,vmax=1)
    ax.set_yticks([0,8,16,23,31,39],labels = [1,0.6,0.2,-0.2,-0.6,-1],fontsize=10)
    ax.set_xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=10,rotation=90)
    ax.set_xlabel(r'$\tau$ [FWHM]',fontsize=15)
    ax.set_ylabel(r'$\alpha$',fontsize=15)
    
for ax, col in zip(axes[:], [r'$FOM_{\tau}$@20dB', r'$FOM_{\tau}$@10dB', r'$FOM_{\tau}$@0dB']):
    ax.annotate(col, (0.5, 1), xytext=(0, 10), ha='center', va='bottom',
                size=18, xycoords='axes fraction', textcoords='offset points')

fig.legend(loc='outside lower center',handles=[OMPPatch,SDPatch,MUSICPatch,ARPatch],ncol=4,fontsize=15,borderaxespad=0.02)
plt.savefig('Maps/Maps_THz_Best_Tau.png',dpi=600)

# Maps Best Alpha

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True,constrained_layout=True)
fig.set_size_inches(12, 4)

fig.suptitle(r'THz Data: $FOM_{\alpha}$ Best algorithm vs SNR',fontsize=18)
for i,ax in enumerate(axes.flat):
    im = ax.imshow(FOMBestAlpha[i],aspect='auto',cmap='gray_r',vmin=0,vmax=1)
    ax.set_yticks([0,8,16,23,31,39],labels = [1,0.6,0.2,-0.2,-0.6,-1],fontsize=10)
    ax.set_xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=10,rotation=90)
    ax.set_xlabel(r'$\tau$ [FWHM]',fontsize=15)
    ax.set_ylabel(r'$\alpha$',fontsize=15)
    
for ax, col in zip(axes[:], [r'$FOM_{\alpha}$@20dB', r'$FOM_{\alpha}$@10dB', r'$FOM_{\alpha}$@0dB']):
    ax.annotate(col, (0.5, 1), xytext=(0, 10), ha='center', va='bottom',
                size=18, xycoords='axes fraction', textcoords='offset points')

fig.legend(loc='outside lower center',handles=[OMPPatch,SDPatch,MUSICPatch,ARPatch],ncol=4,fontsize=15,borderaxespad=0.02)

plt.savefig('Maps/Maps_THz_Best_Alpha.png',dpi=600)

#%% Ranking
Total = len(alpha)*len(lam)

ErrorFOMTau = np.array([np.count_nonzero(FOMDelay[i]>1) for i in range(12)])
ErrorFOMAlpha = np.array([np.count_nonzero(FOMRatio[i]>1) for i in range(12)])

OMPErrorTau = np.array(ErrorFOMTau[[0,4,8]])
OMPErrorAlpha = np.array(ErrorFOMAlpha[[0,4,8]])

SDErrorTau = np.array(ErrorFOMTau[[1,5,9]])
SDErrorAlpha = np.array(ErrorFOMAlpha[[1,5,9]])

MUSICErrorTau = np.array(ErrorFOMTau[[2,6,10]])
MUSICErrorAlpha = np.array(ErrorFOMAlpha[[2,6,10]])

ARErrorTau = np.array(ErrorFOMTau[[3,7,11]])
ARErrorAlpha = np.array(ErrorFOMAlpha[[3,7,11]])

def Rank(name,ErrorTau,ErrorAlpha):
    Fails = (np.sum(ErrorTau)+np.sum(ErrorAlpha))/(6*Total)*100
    Tau = np.mean(ErrorTau/Total*100)
    Alpha = np.mean(ErrorAlpha/Total*100)
    RobustnessTau = (0.2*(ErrorTau[0]/Total*100)+
                  0.3*(ErrorTau[1]/Total*100)+
                  0.5*(ErrorTau[2]/Total*100))
    RobustnessAlpha = (0.2*(ErrorAlpha[0]/Total*100)+
                  0.3*(ErrorAlpha[1]/Total*100)+
                  0.5*(ErrorAlpha[2]/Total*100))
    
    print(f'{name} Fails: {np.round(Fails,1)}% or {np.rint(np.round(Fails)/20)}')
    print(f'{name} FOM_t: {np.round(100-Tau,1)}% or {np.rint(np.round(100-Tau)/20)}')
    print(f'{name} FOM_a: {np.round(100-Alpha,1)}% or {np.rint(np.round(100-Alpha)/20)}')
    print(f'{name} Robustness_t: {np.round(100-RobustnessTau,1)}% or {np.rint(np.round(100-RobustnessTau)/20)}')
    print(f'{name} Robustness_a: {np.round(100-RobustnessAlpha,1)}% or {np.rint(np.round(100-RobustnessAlpha)/20)}\n')
    pass
    
OMPRank = Rank('OMP',OMPErrorTau,OMPErrorAlpha)
SDRank = Rank('SD',SDErrorTau,SDErrorAlpha)
MUSICRank = Rank('MUSIC',MUSICErrorTau,MUSICErrorAlpha)
ARRank = Rank('AR',ARErrorTau,ARErrorAlpha)

#%%
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(ErrorAlpha1[i],aspect='auto',vmin=0,vmax=1,cmap='gray_r')
