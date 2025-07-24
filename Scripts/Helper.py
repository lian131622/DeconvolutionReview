import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert,find_peaks

def Normalize(Reference, verbose=False):
    Norm = Reference/np.max(np.abs(Reference))
    # Plot Reference
    if verbose:
        plt.plot(Norm)
        plt.xlabel('Amplitude')
        plt.ylabel('# of samples')
        plt.title('Reference')
        plt.show()
    return Norm

def GenerateSignal(Reference,alpha,tau):
    # Padding Reference and compute FFT
    Nw = len(Reference)
    if (2*Nw+Nw//2)%2 == 0:
        ReferencePad = np.r_[np.zeros(Nw), Reference, np.zeros(Nw//2)]
    else:
        ReferencePad = np.r_[np.zeros(Nw), Reference, np.zeros((Nw//2)+1)]
    FirstPeak = np.fft.fft(ReferencePad)

    # Create the second peak and shift of tau sample in frequency domain
    SecondPeak = np.fft.fft(alpha*ReferencePad)
    N = len(FirstPeak)
    if N % 2 == 1:
        shift = np.exp(-2j*np.pi*np.arange(1, (N+1)//2)/N*tau)
        SecondPeak[1:(N+1)//2] = SecondPeak[1:(N+1)//2]*shift
        SecondPeak[(N+1)//2:] = np.conj(np.flip(SecondPeak[1:(N+1)//2]))
    else:
        shift = np.exp(-2j*np.pi*np.arange(1, (N//2))/N*tau)
        SecondPeak[1:(N//2)] = SecondPeak[1:(N//2)]*shift
        SecondPeak[(N//2+1):] = np.conj(np.flip(SecondPeak[1:(N//2)]))

    return np.fft.ifft(FirstPeak+SecondPeak).real

def AddNoise(Signal, SNR_dB):
    # Convert SNR from dB to linear scale
    SNR_linear = 10**(SNR_dB/10)

    # Calculate the power of the original signal
    SignalPower = np.mean(Signal**2)

    # Calculate the power of the noise to achieve the desired SNR
    NoisePower = SignalPower/SNR_linear

    # Generate random noise with the specified power
    Noise = np.sqrt(NoisePower)*np.random.normal(0, 1, len(Signal))

    return Signal+Noise

def FWHM(Reference):
    ReferenceH = hilbert(Reference)
    ReferenceEnv = np.abs(ReferenceH)
    MaxEnv = np.max(ReferenceEnv)
    taus = np.argwhere(ReferenceEnv >= 0.5*MaxEnv)
    return abs((taus[-1]-taus[0]).squeeze())

# This is a draft
def OptimizedFindPeaks(Signal,N):
    
    # Find all peaks
    PositiveTaus, PositivePh = find_peaks(Signal, height=0)
    NegativeTaus,NegativePh = find_peaks(-Signal, height=0)
    
    # Sort peaks and valleys by amplitude
    PositivePhSorted = np.flip(np.sort(PositivePh['peak_heights']))
    NegativePhSorted = np.flip(np.sort(NegativePh['peak_heights']))
    
    # Find index of sorted peaks and valleys
    PositiveIdxSorted = np.flip(np.argsort(PositivePh['peak_heights']))
    NegativeIdxSorted = np.flip(np.argsort(NegativePh['peak_heights']))
    
    # Select N peaks and valleys
    PTaus = PositiveTaus[PositiveIdxSorted[:N]]
    NTaus = NegativeTaus[NegativeIdxSorted[:N]]
    
    # Create a matrix where:
        #First row contain the amplitude of the peaks
        #Second row contain the amplitude of the valleys
    PhPN = np.vstack([PositivePhSorted[:N],
                      NegativePhSorted[:N]])

    taus = []
    amp = []

    for i in range(N):
        PeaksValleys = PhPN[0,i]>PhPN[1,i]

        if PeaksValleys == True:
            taus.append(PTaus[i])
            amp.append(PositivePhSorted[i])
            PhPN[1,:] = np.roll(PhPN[1,:],1)
            NTaus = np.roll(NTaus,1)
            NegativePhSorted = np.roll(NegativePhSorted,1)
        else:
            taus.append(NTaus[i])
            amp.append(-NegativePhSorted[i])
            PhPN[0,:] = np.roll(PhPN[0,:],1)
            PTaus = np.roll(PTaus,1)
            PositivePhSorted = np.roll(PositivePhSorted,1)

    return taus,amp

def FindKnee(x,y):
    
    # Fit a straight line from the first to the last point
    m = (y[-1]-y[0])/(x[-1]-x[0])
    b = y[0]-m*x[0]
    
    # Calculate the perpendicular distance from each point to the line
    Distances = np.abs(m*x+b-y)/np.sqrt((m**2)+1)
    
    # Find the index of the maximum distance
    KneeIdx = np.argmax(Distances)
    
    return KneeIdx
