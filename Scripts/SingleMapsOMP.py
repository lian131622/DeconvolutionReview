import Helper
import numpy as np
import tqdm
import time
from scipy.io import loadmat
from Deconvolution import GreedyAlgorithms

path = './Data/NewTHz.mat'

SNR_dB = 20
SAVE_PATH = f'./Maps/DataTHz/OMP_NewTHz_{SNR_dB}dB.mat'

# Import Reference and generate benchmark
Reference = loadmat(path)['ref'].squeeze()
Reference = Helper.Normalize(Reference)

Subsample = 1
Reference = Reference[::Subsample]
FWHM = Helper.FWHM(Reference)


alpha = np.flip(np.r_[np.arange(-1,0,0.05),np.arange(0.05,1.05,0.05)])
lam = np.arange(0.1,1.05,0.05)

# alpha = np.flip(np.r_[np.arange(-1,0,0.2),np.arange(0.05,1.05,0.2)])
# lam = np.arange(0.1,1.05,0.2)

RayleighSignal = Helper.GenerateSignal(Reference,1,1*FWHM)

#Parameters
Ns = len(RayleighSignal)
Nw = len(Reference)


AlphaS = []
TauS = []
Impulses = []

for a in alpha:
    for l in lam:
        AlphaS.append([1,a])
        TauS.append([Nw,Nw+l*FWHM])
        
        h = np.zeros(Ns)
        h[Nw] = 1
        h[int(Nw+l*FWHM)] = a
        Impulses.append(h)

AlphaS = np.array(AlphaS)
TauS = np.array(TauS)

#%% Run Algorithms
Time = []
AlphaMaps = []
TauMaps = []
SignalMaps = []
SignalRMaps = []
NMean = 1

for _ in tqdm.tqdm(range(NMean)):
    
    #Generate Benchmark
    Signals = []
    for a in alpha:
        for l in lam:
            H = Helper.GenerateSignal(Reference,a,l*FWHM)
            H = Helper.AddNoise(H,SNR_dB)
            Signals.append(H)
    
    Signals = np.stack(Signals)
    SignalMaps.append(Signals)
    
    # Start simulation
    AlphaR = []
    TauR = []   
    SignalR = []
    for i in range(len(Signals)):
        
        # Algorithms 
        start = time.time()
        aOMP,tOMP,SignalOMP = GreedyAlgorithms.OptimizedOMP(Signals[i],Reference,2)
        stop = time.time()
        Time.append(stop-start)
        
        # Save Data
        TauR.append(tOMP)
        AlphaR.append(aOMP)
        SignalR.append(SignalOMP)
     
    # Keep data for Maps
    TauMaps.append(np.array(TauR))
    AlphaMaps.append(np.array(AlphaR))
    SignalRMaps.append(np.array(SignalR))

# Make lists arrays
SignalMaps = np.array(SignalMaps)
SignalRMaps = np.array(SignalRMaps)
AlphaMaps = np.array(AlphaMaps)
TauMaps = np.array(TauMaps)

# Compute the mean
TauMapsMean = np.mean(TauMaps,axis=0)
AlphaMapsMean = np.mean(AlphaMaps,axis=0)

print(f'Elapsed time: {np.sum(np.array(Time))*1e3} ms')
print(f'Mean Time of algorithm: {np.mean(np.array(Time))*1e3} ms')
#%% FOM Time of Flight
FOM_FirstToF = (np.abs(TauMapsMean[:,0]-TauS[:,0])/FWHM).reshape(len(alpha),len(lam))
FOM_SecondToF = (np.abs(TauMapsMean[:,1]-TauS[:,1])/FWHM).reshape(len(alpha),len(lam))

DelayS = np.abs(TauS[:,1]-TauS[:,0])
DelayR = np.abs(TauMapsMean[:,1]-TauMapsMean[:,0])
FOM_Delay = (np.abs(DelayS-DelayR)/DelayS).reshape(len(alpha),len(lam))

# FOM Amplitude
FOM_FirstPeak = (np.abs(AlphaMapsMean[:,0]-AlphaS[:,0])).reshape(len(alpha),len(lam))
FOM_SecondPeak = (np.abs(AlphaMapsMean[:,1]-AlphaS[:,1])).reshape(len(alpha),len(lam))

RatioS = AlphaS[:,1]/AlphaS[:,0]
RatioR = AlphaMapsMean[:,1]/AlphaMapsMean[:,0]
FOM_Ratio = (np.abs(RatioS-RatioR)/np.abs(RatioS)).reshape(len(alpha),len(lam))

#%%
import matplotlib.pyplot as plt

plt.figure()

plt.imshow(FOM_Delay,vmin=0,vmax=1,cmap='gray_r',aspect='auto')
plt.yticks(np.arange(len(alpha))[::3],labels = np.round(alpha,2)[::3],fontsize=8)
plt.xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=8,rotation=90)
plt.show()

#%% Save Data
from scipy.io import savemat

mdic = {'SNR_dB': SNR_dB, 'alpha': alpha, 'lam':lam,'NMean':NMean,
        'FWHM':FWHM,'AlphaS': AlphaS, 'TauS':TauS, 'Impulses':Impulses,
        'SignalMaps': SignalMaps,'SignalRMaps':SignalRMaps,
        'AlphaMaps':AlphaMaps,'TauMaps':TauMaps}

savemat(SAVE_PATH,mdic)

#%%
NSearch = 2
plt.figure()
plt.imshow(FOM_Delay,vmin=0,vmax=1,cmap='gray_r',aspect='auto')
plt.yticks(np.arange(len(alpha))[::3],labels = np.round(alpha,2)[::3],fontsize=8)
plt.xticks(np.arange(len(lam))[::3],labels = np.round(lam,2)[::3],fontsize=8,rotation=90)
Coord = plt.ginput(NSearch)

Coord = np.array(Coord)
x = np.ceil(Coord[:,0])
y = np.ceil(Coord[:,1])
Position = np.uint32(((y+1)*(x+1))-1)

fig,ax = plt.subplots(NSearch,1,constrained_layout=True)
for i in range(NSearch):
    ax[i].plot(SignalRMaps[0,Position[i],:])
plt.show()