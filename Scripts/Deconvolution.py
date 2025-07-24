import numpy as np
import Helper
from scipy.linalg import toeplitz
from scipy.signal import find_peaks,windows
from spectrum import arburg

class GenericAlgorithms():
    def WienerDeconvolution(Signal, Reference):
        # Parameters
        Ns = len(Signal)
        Nw = len(Reference)
    
        # FFT
        In = np.fft.fft(np.r_[Reference, np.zeros(Ns-Nw)])
        Out = np.fft.fft(Signal)
    
        # IFFT
        SignalW = np.fft.ifft(Out*In.conj()/(np.abs(In)**2 + (0.01*np.max(np.abs(In)**2)))).real
    
        return SignalW

class AutoRegressiveAlgorithms():
    def ARSE(Signal, Reference, N, BW=0.5):

        #Parameters
        Ns = len(Signal)
        
        #FFT
        Y = np.fft.fft(Signal)
        W = np.fft.fft(Reference,Ns)

        MaxFFT = np.max(np.abs(W)[:Ns//2])
        idx = np.argwhere(np.abs(W)[:Ns//2] >= BW*MaxFFT).squeeze()
        iL = int(idx[0])
        iH = int(idx[-1])

        H = Y[iL:iH]/W[iL:iH]
        # H = fft(GenericAlgorithms.WienerDeconvolution(Signal, Reference))

        #Compute the coefficient
        p = np.rint((iH-iL)/2).astype(np.uint32)

        a, rho, ref = arburg(H, p)
        
        # Make the coefficients stable
        poles = np.roots(np.r_[1, a])
        poles_stable = np.array([p/(np.abs(p)*np.abs(p)) if np.abs(p) > 1 else p for p in poles])

        a_stable = np.poly(poles_stable)
        a_stable = a_stable[1:]

        a = a_stable
        
        #Reconstruction of signal
        HalfH_AR = np.zeros(Ns//2, dtype='complex')
        HalfH_AR[iL:iH] = H
        
        for i in range(iL-1,-1,-1):
            HalfH_AR[i] = -np.sum(np.conj(a)*HalfH_AR[i+1:i+p+1])
            
        for i in range(iH,Ns//2):
            HalfH_AR[i] = -np.sum(a*HalfH_AR[i-1:i-p-1:-1])
            
        H_AR = np.r_[HalfH_AR, np.conj(np.flip(HalfH_AR[1:]))]
        
        WindowsHann = np.fft.fftshift(windows.hann(len(H_AR)))
        
        H_ARFiltered = H_AR*WindowsHann
        AR = np.fft.ifft(H_ARFiltered).real
        
        AR = np.r_[0,AR] # Just for our benchmark

        # Find peaks of signal and sort
        # ARh = np.abs(hilbert(AR))
        # taus, Ph = find_peaks(ARh,height=0)
        ARAbs = np.abs(AR)
        taus, Ph = find_peaks(ARAbs,height=0)
        IdxSorted = np.flip(np.argsort(Ph['peak_heights']))
        taus = np.sort(taus[IdxSorted[:N]])
        
        tausOpt = []
        for t in taus:
            deltaDelay = (ARAbs[t-1]-ARAbs[t+1])/(2*(ARAbs[t-1]-2*ARAbs[t]+ARAbs[t+1]))
            # deltaDelay = (ARh[t-1]-ARh[t+1])/(2*(ARh[t-1]-2*ARh[t]+ARh[t+1]))
            tausOpt.append(t+deltaDelay)
            
        # amp = np.flip(np.sort(Ph['peak_heights']))[:N]
        amp = AR[np.rint(tausOpt).astype(np.uint32)]
        
        return amp, np.sort(tausOpt), AR
    
    def OptimizedARSE(Signal, Reference, N):

        #Parameters
        Ns = len(Signal)

        #FFT
        Y = np.fft.fft(Signal)
        W = np.fft.fft(Reference,Ns)
        BW = [0.7,0.5,0.3]
        MaxFFT = np.max(np.abs(W)[:Ns//2])

        H_AR = []
        p = 0
        for k in range(0,len(BW)):

            idx = np.argwhere(np.abs(W)[:Ns//2] >= BW[k]*MaxFFT).squeeze()
            iL = int(idx[0])
            iH = int(idx[-1])

            H = Y[iL:iH]/W[iL:iH]
            # H = fft(GenericAlgorithms.WienerDeconvolution(Signal, Reference))
            
            if k == 0:
                #Compute the coefficient
                p = np.rint((iH-iL)/2).astype(np.uint32)
            else:
                p = p
                
            a, _, _ = arburg(H, p)
            
            # Make the coefficients stable
            poles = np.roots(np.r_[1, a])
            poles_stable = np.array([p/(np.abs(p)*np.abs(p)) if np.abs(p) > 1 else p for p in poles])

            a_stable = np.poly(poles_stable)
            a_stable = a_stable[1:]

            a = a_stable
            
            #Reconstruction of signal
            HalfH_AR = np.zeros(Ns//2, dtype='complex')
            HalfH_AR[iL:iH] = H
            
            for i in range(iL-1,-1,-1):
                HalfH_AR[i] = -np.sum(np.conj(a)*HalfH_AR[i+1:i+p+1])
                
            for i in range(iH,Ns//2):
                HalfH_AR[i] = -np.sum(a*HalfH_AR[i-1:i-p-1:-1])
            
            H_AR.append(np.r_[HalfH_AR,np.conj(np.flip(HalfH_AR[1:]))])

        H_ARMean = np.mean(np.array(H_AR),axis=0)

        WindowsHann = np.fft.fftshift(windows.hann(len(H_ARMean)))
        
        H_ARFiltered = H_ARMean*WindowsHann
        AR = np.fft.ifft(H_ARFiltered).real
        
        AR = np.r_[0,AR] # Just for our benchmark

        # Find peaks of signal and sort
        ARAbs = np.abs(AR)
        # ARh = np.abs(hilbert(AR))
        taus, Ph = find_peaks(ARAbs, height=0)
        IdxSorted = np.flip(np.argsort(Ph['peak_heights']))
        taus = np.sort(taus[IdxSorted[:N]])

        tausOpt = []
        for t in taus:
            deltaDelay = (ARAbs[t-1]-ARAbs[t+1])/(2*(ARAbs[t-1]-2*ARAbs[t]+ARAbs[t+1]))
            # deltaDelay = (ARh[t-1]-ARh[t+1])/(2*(ARh[t-1]-2*ARh[t]+ARh[t+1]))
            tausOpt.append(t+deltaDelay)
            
        if len(tausOpt) < N:
            tausOpt.append(np.repeat(Ns-1,N-len(tausOpt)).squeeze())
            
        # amp = np.flip(np.sort(Ph['peak_heights']))[:N]
        amp = AR[np.rint(tausOpt).astype(np.uint32)]
        
        return amp, np.sort(tausOpt), AR
    
    def ARSEPinv(Signal, Reference, N):
        Ns = len(Signal)
        Nw = len(Reference)

        OldAmp,ARTau,SignalAR = AutoRegressiveAlgorithms.OptimizedARSE(Signal, Reference, N)
        
        ampOpt = []
        
        # Shift in frequency domain
        M = np.zeros((Ns,N))

        for i in range(len(ARTau)):
            ReferencePad = np.r_[Reference,np.zeros(Ns-Nw)]
            FFTReplica = np.fft.fft(ReferencePad)
            Nr = len(FFTReplica)
            
            if Nr%2 == 1:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr+1)//2)/Nr*(ARTau[i]))
                FFTReplica[1:(Nr+1)//2] = FFTReplica[1:(Nr+1)//2]*shift
                FFTReplica[(Nr+1)//2:] = np.conj(np.flip(FFTReplica[1:(Nr+1)//2]))
            else:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr//2))/Nr*(ARTau[i]))
                FFTReplica[1:(Nr//2)] = FFTReplica[1:(Nr//2)]*shift
                FFTReplica[(Nr//2+1):] = np.conj(np.flip(FFTReplica[1:(Nr//2)]))
            
            # Reconstruct amplitude
            M[:,i] = np.fft.ifft(FFTReplica).real
                
        ampOpt = np.linalg.pinv(M)@Signal
        
        return ampOpt, np.sort(ARTau), SignalAR
        
class ISTAlgorithms():
    def L1(Signal, Reference, N, mu, lam=0.5, N_iter=200, eps=1e-3):
    
        #Parameters
        Ns = len(Signal)
        Nw = len(Reference)

        #Create Toeplitz matrix
        X = toeplitz(np.r_[Reference, np.zeros(Ns-Nw)],
                     np.r_[Reference[0], np.zeros(Ns-1)])

        hk = np.zeros(Ns)
        
        #ISTA algorithm
        for _ in range(N_iter):
            # Gradient descent step
            hkk = hk - mu*X.T@(X@hk - Signal)
            
            # Soft-thresholding for L1 regularization
            hkk = np.sign(hkk)*np.maximum(np.abs(hkk)-lam*mu,0)
            
            if np.linalg.norm(hkk-hk,2) < eps:
                break
            hk = hkk
        
        # hk = np.roll(hk,idxPhi)
        # taus, Ph = find_peaks(np.abs(hilbert(hk)), height=0)
        hkAbs = np.abs(hk)
        taus, Ph = find_peaks(hkAbs, height=0)
        IdxSorted = np.flip(np.argsort(Ph['peak_heights']))
        taus = np.sort(taus[IdxSorted[:N]])
        # amp = np.flip(np.sort(Ph['peak_heights']))[:N]
        
        tausOpt = []
        for t in taus:
            deltaDelay = (hkAbs[t-1]-hkAbs[t+1])/(2*(hkAbs[t-1]-2*hkAbs[t]+hkAbs[t+1]))
            tausOpt.append(t+deltaDelay)
            
        if len(tausOpt) < N:
            tausOpt.append(np.repeat(Ns-1,N-len(tausOpt)).squeeze())
        
        amp = hk[np.rint(tausOpt).astype(np.uint32)]
        
        return amp, tausOpt, hk
    
    def OptimizedL1(Signal, Reference, N, mu, beta=1, N_iter=200,eps=1e-6):
    
        #Parameters
        Ns = len(Signal)
        Nw = len(Reference)

        #Create Toeplitz matrix
        X = toeplitz(np.r_[Reference, np.zeros(Ns-Nw)],
                     np.r_[Reference[0], np.zeros(Ns-1)])

        lam = np.geomspace(1e-2,1e2,50)
        zk = np.zeros((Ns,len(lam)))
        ResidualError = np.zeros(len(lam))
        RegularizationError = np.zeros(len(lam))
        LCurve = []
        j = 0

        for l in lam:
            hk = np.zeros(Ns)
            #ISTA algorithm
            for _ in range(N_iter):
                # Gradient descent step
                hkk = hk - mu*X.T@(X@hk - Signal)
                
                # Soft-thresholding for L1 regularization
                hkk = np.sign(hkk)*np.maximum(np.abs(hkk)-l*mu,0)
                
                if np.linalg.norm(hkk-hk,2) < eps:
                    break
                
                hk = hkk

            zk[:,j] = hk
            ResidualError[j] = np.sum(np.abs(X@zk[:,j]-Signal)**2)
            RegularizationError[j] = np.sum(np.abs(zk[:,j]))
            j = j+1
        
        LCurve = np.stack([ResidualError,RegularizationError])
        IdxL = Helper.FindKnee(ResidualError,RegularizationError)
        NewIdxL = np.argwhere(lam<=beta*lam[IdxL]).squeeze()[-1]

        hk = zk[:,NewIdxL]
        
        hkAbs = np.abs(hk)
        taus, Ph = find_peaks(hkAbs, height=0)
        IdxSorted = np.flip(np.argsort(Ph['peak_heights']))
        taus = np.sort(taus[IdxSorted[:N]])

        tausOpt = []
        for t in taus:
            deltaDelay = (hkAbs[t-1]-hkAbs[t+1])/(2*(hkAbs[t-1]-2*hkAbs[t]+hkAbs[t+1]))
            tausOpt.append(t+deltaDelay)  
            
        if len(tausOpt) < N:
            tausOpt.append(np.repeat(Ns-1,N-len(tausOpt)))
        
        amp = hk[np.rint(tausOpt).astype(np.uint32)]
        
        return amp, tausOpt, hk, lam, zk, LCurve
    
    def L1Pursuit(Signal, Reference, N, mu, lam=0.5):
        Ns = len(Signal)
        Nw = len(Reference)
        
        Delay = np.argmax(np.abs(Reference))

        ReferencePad = np.roll(np.r_[Reference,np.zeros(Ns-Nw)],Delay)
        _,_,ReferenceL1 = ISTAlgorithms.L1(ReferencePad, Reference, N, mu, lam)
        
        _,_,SignalL1 = ISTAlgorithms.L1(Signal, Reference, N, mu, lam)
        
        aL1Pur,tL1Pur,SignalL1Pur = GreedyAlgorithms.OptimizedOMP(SignalL1,ReferenceL1,N)
        
        return aL1Pur,np.sort(tL1Pur+Delay),np.roll(SignalL1Pur,Delay)
    
    def L1Pinv(Signal, Reference, N, mu, lam):
        Ns = len(Signal)
        Nw = len(Reference)
        
        OldAmp,tL1,SignalL1 = ISTAlgorithms.L1(Signal, Reference, N, mu, lam)
        
        ampOpt = []
        
        # Shift in frequency domain   
        M = np.zeros((Ns,N))

        for i in range(len(tL1)):
            ReferencePad = np.r_[Reference,np.zeros(Ns-Nw)]
            FFTReplica = np.fft.fft(ReferencePad)
            Nr = len(FFTReplica)
            
            if Nr%2 == 1:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr+1)//2)/Nr*(tL1[i]))
                FFTReplica[1:(Nr+1)//2] = FFTReplica[1:(Nr+1)//2]*shift
                FFTReplica[(Nr+1)//2:] = np.conj(np.flip(FFTReplica[1:(Nr+1)//2]))
            else:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr//2))/Nr*(tL1[i]))
                FFTReplica[1:(Nr//2)] = FFTReplica[1:(Nr//2)]*shift
                FFTReplica[(Nr//2+1):] = np.conj(np.flip(FFTReplica[1:(Nr//2)]))
            
            # Reconstruct amplitude
            M[:,i] = np.fft.ifft(FFTReplica).real
                
        ampOpt = np.linalg.pinv(M)@Signal
        
        return ampOpt,np.sort(tL1),SignalL1

class GreedyAlgorithms():
    def OMP(Signal, Reference, N):
    
        # Parameters
        Ns = len(Signal)
        Nw = len(Reference)

        # Create Toeplitz matrix
        ReferencePad = np.r_[Reference, np.zeros(Ns-Nw)]

        # Delay 
        phiRef = np.correlate(Reference,Reference,mode='full')
        Delay = np.argmax(np.abs(phiRef))

        res = Signal
        taus = []
        M = np.zeros((len(Signal),N))
        for i in range(0, N):
            MP = np.convolve(np.flip(Reference),res,mode='full')
            idxTau = np.argmax(np.abs(MP))
            deltaDelay = (MP[idxTau-1]-MP[idxTau+1])/(2*(MP[idxTau-1]-2*MP[idxTau]+MP[idxTau+1]))
            taus.append(idxTau-Delay+deltaDelay)
            M[:,i] = np.roll(ReferencePad,int(np.round(taus[i])))
            if taus[i] > Ns-Nw+Delay:
                M[0:taus[i]+Nw-Delay-Ns,i] = 0
            amp = np.linalg.pinv(M[:,:i+1])@Signal
            res = Signal - M[:,:i+1]@amp

        SignalR = np.zeros(Ns)
        SignalR[np.rint(taus).astype(np.uint32)] = amp
    
        return amp, taus, SignalR
    
    def OptimizedOMP(Signal, Reference, N):
        # Parameters
        Ns = len(Signal)
        Nw = len(Reference)

        # Parabolic interpolation to estimate subsample delay
        phiRef = np.correlate(Reference,Reference,mode='full')
        Delay = np.argmax(np.abs(phiRef))

        # Initialize
        tausOpt = []
        res = Signal
        M = np.zeros((Ns,N))

        for i in range(0, N):
            MP = np.convolve(np.flip(Reference),res,mode='full')
            idxTau = np.argmax(np.abs(MP))
            deltaTau = (MP[idxTau-1]-MP[idxTau+1])/(2*(MP[idxTau-1]-2*MP[idxTau]+MP[idxTau+1]))
            tausOpt.append(idxTau-Delay+deltaTau)
            if tausOpt[i] < 0:
                tausOpt[i] = 0
                
            # Shift in frequency domain
            ReferencePad = np.r_[Reference,np.zeros(Ns-Nw)]
            FFTReplica = np.fft.fft(ReferencePad)
            Nr = len(FFTReplica)
            if Nr%2 == 1:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr+1)//2)/Nr*(tausOpt[i]))
                FFTReplica[1:(Nr+1)//2] = FFTReplica[1:(Nr+1)//2]*shift
                FFTReplica[(Nr+1)//2:] = np.conj(np.flip(FFTReplica[1:(Nr+1)//2]))
            else:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr//2))/Nr*(tausOpt[i]))
                FFTReplica[1:(Nr//2)] = FFTReplica[1:(Nr//2)]*shift
                FFTReplica[(Nr//2+1):] = np.conj(np.flip(FFTReplica[1:(Nr//2)]))
            
            # Reconstruct amplitude
            M[:,i] = np.fft.ifft(FFTReplica).real
                
            ampOpt = np.linalg.pinv(M[:,:i+1])@Signal
            res = Signal - M[:,:i+1]@ampOpt
            
        SignalR = np.zeros(Ns)
        SignalR[np.rint(tausOpt).astype(np.uint32)] = ampOpt
        
        return ampOpt, np.sort(tausOpt), SignalR

class SubspaceAlgorithms():
    def MUSIC(Signal, Reference, N, BW=0.1):
        Ns = len(Signal)
        Nw = len(Reference)
    
        #FFT
        Y = np.fft.fft(Signal)
        W = np.fft.fft(Reference,Ns)
    
        MaxFFT = np.max(np.abs(W)[:Ns//2])
        idx = np.argwhere(np.abs(W)[:Ns//2] >= BW*MaxFFT).squeeze()
        iL = int(idx[0])
        iH = int(idx[-1])
        
        Ncorr = (iH-iL)//2
        
        H = Y[iL:iH]/W[iL:iH]

        #Choose N correlation matrix and average
        Rxx = np.zeros((iH-iL-Ncorr,iH-iL-Ncorr), dtype='complex')
        for i in range(Ncorr):
            S = H[i:-Ncorr+i].reshape(-1, 1)
            Rxx = Rxx + np.outer(S,S.conj())

        Rxx = Rxx/Ncorr

        N_FFT = iH-iL-Ncorr
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Rxx)
        Q = eigenvectors[:, N:]
    
        # Compute pseudospectrum
        tau = np.arange(0, Ns)
        Step = -2*1j*np.pi*np.arange(0,N_FFT)/Ns
        v = np.exp(np.outer(tau,Step))
        P = 1/np.linalg.norm(v@Q.conj(),axis=1)**2
        
        # Find peaks of pseudospectrum and sort
        taus, Ph = find_peaks(P,height=0)
        IdxSorted = np.flip(np.argsort(Ph['peak_heights']))
        taus = taus[IdxSorted[:N]]

        ReferencePad = np.r_[Reference, np.zeros(Ns-Nw)]
    
        M = np.zeros((Ns,N))
        for i in range(N):
            M[:,i] = np.roll(ReferencePad,(taus[i]))
    
        amp = np.linalg.pinv(M)@Signal
        
        # Reconstruction of Signal
        SignalR = np.zeros(Ns)
        SignalR[taus] = amp
        
        return amp, np.sort(taus), SignalR
    
    def NMUSIC(Signal,Reference,N,BW=0.1):
        Ns = len(Signal)
        Nw = len(Reference)
    
        #FFT
        Y = np.fft.fft(Signal)
        W = np.fft.fft(Reference,Ns)
    
        MaxFFT = np.max(np.abs(W)[:Ns//2])
        idx = np.argwhere(np.abs(W)[:Ns//2] >= BW*MaxFFT).squeeze()
        iL = int(idx[0])
        iH = int(idx[-1])
        
        Ncorr = (iH-iL)//2
        
        H = Y[iL:iH]/W[iL:iH]

        #Choose N correlation matrix and average
        Rxx = np.zeros((iH-iL-Ncorr,iH-iL-Ncorr), dtype='complex')
        for i in range(Ncorr):
            S = H[i:-Ncorr+i].reshape(-1, 1)
            Rxx = Rxx + np.outer(S,S.conj())

        Rxx = Rxx/Ncorr

        N_FFT = iH-iL-Ncorr
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(Rxx)
        Idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[Idx]
        eigenvectors = eigenvectors[:,Idx]
        
        # Compute pseudospectrum
        
        M = np.zeros((Ns,N))
        tausOpt = []
        for i in range(N):
            Q = eigenvectors[:,i]
            Ps = np.outer(Q,Q.conj())
            tau = np.arange(0, Ns)
            Step = -2*1j*np.pi*np.arange(0,N_FFT)/Ns
            v = np.exp(np.outer(tau,Step))
            P = np.linalg.norm(Ps@v.T,axis=0)**2
            tau = np.argmax(P)
            
            if tau == 0:
                tausOpt.append(tau)
                
            elif tau == Ns-1:
                tausOpt.append(tau)
                
            else:
                deltaDelay = (P[tau-1]-P[tau+1])/(2*(P[tau-1]-2*P[tau]+P[tau+1]))
                tausOpt.append(tau+deltaDelay)
            
            ReferencePad = np.r_[Reference,np.zeros(Ns-Nw)]
            FFTReplica = np.fft.fft(ReferencePad)
            Nr = len(FFTReplica)
            
            if Nr%2 == 1:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr+1)//2)/Nr*(tausOpt[i]))
                FFTReplica[1:(Nr+1)//2] = FFTReplica[1:(Nr+1)//2]*shift
                FFTReplica[(Nr+1)//2:] = np.conj(np.flip(FFTReplica[1:(Nr+1)//2]))
            else:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr//2))/Nr*(tausOpt[i]))
                FFTReplica[1:(Nr//2)] = FFTReplica[1:(Nr//2)]*shift
                FFTReplica[(Nr//2+1):] = np.conj(np.flip(FFTReplica[1:(Nr//2)]))
            
            # Reconstruct amplitude
            M[:,i] = np.fft.ifft(FFTReplica).real
                
        ampOpt = np.linalg.pinv(M)@Signal
        
        # Reconstruction of Signal
        SignalR = np.zeros(Ns)
        SignalR[np.rint(tausOpt).astype(np.uint32)] = ampOpt
            
        return ampOpt,np.sort(tausOpt),SignalR

    def OptimizedMUSIC(Signal, Reference, N, Resolution=1):
        Ns = len(Signal)
        Nw = len(Reference)

        BW = [0.3,0.5,0.7]
        P = []
        gamma = []
        for k in range(0,len(BW)):
            #FFT
            Y = np.fft.fft(Signal)
            W = np.fft.fft(Reference,Ns)

            MaxFFT = np.max(np.abs(W)[:Ns//2])
            idx = np.argwhere(np.abs(W)[:Ns//2] >= BW[k]*MaxFFT).squeeze()
            iL = int(idx[0])
            iH = int(idx[-1])

            Ncorr = (iH-iL)//2

            H = Y[iL:iH]/W[iL:iH]

            #Choose N correlation matrix and average
            Rxx = np.zeros((iH-iL-Ncorr,iH-iL-Ncorr), dtype='complex')
            for i in range(Ncorr):
                S = H[i:-Ncorr+i].reshape(-1, 1)
                Rxx = Rxx + np.outer(S,S.conj())

            Rxx = Rxx/Ncorr

            N_FFT = iH-iL-Ncorr

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(Rxx)
            Q = eigenvectors[:, N:]

            # Compute pseudospectrum
            tau = np.linspace(0,Ns,int(Resolution)*Ns)
            Step = -1j*2*np.pi*np.arange(0,N_FFT)/Ns
            v = np.exp(np.outer(tau,Step))
            P.append(1/(np.linalg.norm(v@Q.conj(),axis=1)**2))
            gamma.append(iH-iL-Ncorr)

        PMean = np.mean(np.array(P),axis=0)
        # PMean = np.sum(np.array(P)*np.array(gamma)[:,None],axis=0)/np.sum(np.array(gamma))

        # Find peaks of pseudospectrum and sort
        taus, Ph = find_peaks(PMean,height=0)
        IdxSorted = np.flip(np.argsort(Ph['peak_heights']))
        taus = np.uint32(taus[IdxSorted[:N]])

        # Shift in frequency domain
        M = np.zeros((Ns,N))
        tausOpt = []
        
        for i in range(len(taus)):
            ReferencePad = np.r_[Reference,np.zeros(Ns-Nw)]
            FFTReplica = np.fft.fft(ReferencePad)
            Nr = len(FFTReplica)
            
            deltaDelay = (PMean[taus[i]-1]-PMean[taus[i]+1])/(2*(PMean[taus[i]-1]-2*PMean[taus[i]]+PMean[taus[i]+1]))
            tausOpt.append(taus[i]+deltaDelay)

            if Nr%2 == 1:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr+1)//2)/Nr*(tausOpt[i]))
                FFTReplica[1:(Nr+1)//2] = FFTReplica[1:(Nr+1)//2]*shift
                FFTReplica[(Nr+1)//2:] = np.conj(np.flip(FFTReplica[1:(Nr+1)//2]))
            else:
                shift = np.exp(-2j*np.pi*np.arange(1,(Nr//2))/Nr*(tausOpt[i]))
                FFTReplica[1:(Nr//2)] = FFTReplica[1:(Nr//2)]*shift
                FFTReplica[(Nr//2+1):] = np.conj(np.flip(FFTReplica[1:(Nr//2)]))
            
            # Reconstruct amplitude
            M[:,i] = np.fft.ifft(FFTReplica).real
                
        ampOpt = np.linalg.pinv(M)@Signal

        if len(tausOpt) < N:
            tausOpt.append(np.repeat(Ns-1,N-len(tausOpt)).squeeze())
        
        # Reconstruction of Signal
        SignalR = np.zeros(Ns)
        SignalR[np.rint(tausOpt).astype(np.uint32)//Resolution] = ampOpt
        
        return ampOpt, np.sort(tausOpt), SignalR