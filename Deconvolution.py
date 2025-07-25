import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import toeplitz

def OMP(Signal, X, N):
    '''

    Parameters
    ----------
    Signal : Measured signal.
    
    X : Dictionary.
    
    N : Number of iterations.

    Returns
    -------
    amp : Vector of amplitudes.
    
    taus : Vector of ToFs.
    
    SignalR : Estimated impulse response.

    '''

    # Initialization
    Ns = len(Signal)
    res = Signal
    taus = []
    
    # Iteration loop
    for _ in range(0, N):
        taus.append(np.argmax(np.abs(res @ X)))
        res = Signal - X[:, taus] @ np.linalg.pinv(X[:, taus]) @ Signal

    # Reconstruct the amplitude
    amp = np.linalg.pinv(X[:, taus]) @ Signal

    # Reconstruct the impulse response
    SignalR = np.zeros(Ns)
    SignalR[np.rint(taus).astype(np.uint32)] = amp

    return amp, taus, SignalR


def SD(Signal, Reference, theta, lam, N_iter=1000, eps=1e-6):
    '''

    Parameters
    ----------
    Signal : Measured signal.
    
    Reference : Reference signal.
    
    theta : Step size controlling the gradient descent step.
    
    lam : Regularization parameter.
    
    N_iter : Number of iteration. The default is 1000.
    
    eps : Stop criterion value. The default is 1e-6.

    Returns
    -------
    hk : Estimated impulse response.

    '''

    # Initialization
    Ns = len(Signal)
    Ns = len(Signal)
    Nw = len(Reference)
    X = toeplitz(np.r_[Reference, np.zeros(Ns-Nw)],np.r_[Reference[0], np.zeros(Ns-1)])

    # ISTA algorithm
    hk = np.zeros(Ns)
    for _ in range(N_iter):
        # Gradient descent step
        hkk = hk - theta * X.T @ (X @ hk - Signal)

        # Soft-thresholding for L1 regularization
        hkk = np.sign(hkk) * np.maximum(np.abs(hkk) - lam * theta, 0)

        # Stop criterion iteration
        if np.linalg.norm(hkk - hk, 2) < eps:
            break
        hk = hkk

    return hk


def MUSIC(Signal, Reference, NSub, BW=0.1):
    '''

    Parameters
    ----------
    Signal : Measured signal.
    
    Reference : Reference signal.
    
    NSub : Dimension of the signal subspace.
        
    BW : [%] of np.max(np.abs(fft(Reference))) to select BW.The default is 0.1.

    Returns
    -------
    amp : Vector of amplitudes.
    
    taus : Vector of ToFs.
    
    SignalR : Estimated impulse response.

    '''

    # Initialization
    Ns = len(Signal)
    Nw = len(Reference)

    # FFT
    Y = np.fft.fft(Signal)
    W = np.fft.fft(Reference, Ns)

    # Select high SNR region
    MaxFFT = np.max(np.abs(W)[:Ns // 2])
    idx = np.argwhere(np.abs(W)[:Ns // 2] >= BW * MaxFFT).squeeze()
    iL = int(idx[0])
    iH = int(idx[-1])
    H = Y[iL:iH] / W[iL:iH]

    # Choose N correlation matrix and average
    Ncorr = (iH - iL) // 2
    Rxx = np.zeros((iH - iL - Ncorr, iH - iL - Ncorr), dtype='complex')
    for i in range(Ncorr):
        S = H[i:-Ncorr + i].reshape(-1, 1)
        Rxx = Rxx + np.outer(S, S.conj())
    Rxx = Rxx / Ncorr

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Rxx)
    Q = eigenvectors[:, NSub:]

    # Compute pseudospectrum
    tau = np.arange(0, Ns)
    Step = -2 * 1j * np.pi * np.arange(0, iH - iL - Ncorr) / Ns
    v = np.exp(np.outer(tau, Step))
    P = 1 / np.linalg.norm(v @ Q.conj(), axis=1) ** 2

    # Find peaks of pseudospectrum
    taus, Ph = find_peaks(P, height=0)

    # Reconstruct the amplitue
    ReferencePad = np.r_[Reference, np.zeros(Ns - Nw)]
    M = np.zeros((Ns, len(taus)))
    for i in range(len(taus)):
        M[:, i] = np.roll(ReferencePad, (taus[i]))
    amp = np.linalg.pinv(M) @ Signal

    # Reconstruct the impulse response
    SignalR = np.zeros(Ns)
    SignalR[taus] = amp

    return amp, taus, SignalR


def arburg(X, order):
    '''
    The program utilizes this function adapted from the library spectrum.
    Cokelaer et al, (2017), 'Spectrum': Spectral Analysis in Python, Journal of Open Source Software, 2(18), 348, doi:10.21105/joss.00348
    
    Parameters
    ----------
    X : Array of complex data samples (length N)
    
    order : Order of autoregressive process (0 < order < N)
    

    Returns
    -------
    a : Array of complex autoregressive parameters A(1) to A(order).
    
    rho : Real variable representing driving noise variance (mean square
      of residual noise) from the whitening operation of the Burg
      filter.
      
    ref : Reflection coefficients defining the filter of the model.

    '''
    
    x = np.array(X)
    N = len(x)

    # Initialisation
    rho = sum(abs(x)**2.) / float(N)
    den = rho * 2. * N


    a = np.zeros(0, dtype=complex)
    ref = np.zeros(0, dtype=complex)
    ef = x.astype(complex)
    eb = x.astype(complex)
    temp = 1.
    
    #   Main recursion
    for k in range(0, order):

        # calculate the next order reflection coefficient Eq 8.14 Marple
        num = sum([ef[j]*eb[j-1].conjugate() for j in range(k+1, N)])
        den = temp * den - abs(ef[k])**2 - abs(eb[N-1])**2
        kp = -2. * num / den #eq 8.14

        temp = 1. - abs(kp)**2.
        new_rho = temp * rho

        # this should be after the criteria
        rho = new_rho

        a.resize(a.size+1,refcheck=False)
        a[k] = kp
        if k == 0:
            for j in range(N-1, k, -1):
                save2 = ef[j]
                ef[j] = save2 + kp * eb[j-1]
                eb[j] = eb[j-1] + kp.conjugate()*save2

        else:
            # update the AR coeff
            khalf = (k+1)//2
            for j in range(0, khalf):
                ap = a[j] # previous value
                a[j] = ap + kp * a[k-j-1].conjugate()
                if j != k-j-1:
                    a[k-j-1] = a[k-j-1] + kp * ap.conjugate()

            # update the prediction error
            for j in range(N-1, k, -1):
                save2 = ef[j]
                ef[j] = save2 + kp * eb[j-1]
                eb[j] = eb[j-1] + kp.conjugate()*save2

        # save the reflection coefficient
        ref.resize(ref.size+1, refcheck=False)
        ref[k] = kp

    return a, rho, ref

def AR(Signal, Reference, p, BW=0.1):
    '''

    Parameters
    ----------
    Signal : Measured signal.
    
    Reference : Reference signal.
    
    p : Model order.
    
    BW : [%] of np.max(np.abs(fft(Reference))) to select BW.The default is 0.1.

    Returns
    -------
    AR : Estimated impulse response.

    '''

    # Initialization
    Ns = len(Signal)

    # FFT
    Y = np.fft.fft(Signal)
    W = np.fft.fft(Reference, Ns)

    # Select high SNR region
    MaxFFT = np.max(np.abs(W)[:Ns // 2])
    idx = np.argwhere(np.abs(W)[:Ns // 2] >= BW * MaxFFT).squeeze()
    iL = int(idx[0])
    iH = int(idx[-1])
    H = Y[iL:iH] / W[iL:iH]

    # Compute the coefficient
    a, rho, ref = arburg(H, p)

    # Reconstruction of the impulse response
    HalfH_AR = np.zeros(Ns // 2, dtype='complex')
    HalfH_AR[iL:iH] = H

    for i in range(iL - 1, -1, -1):
        HalfH_AR[i] = -np.sum(np.conj(a) * HalfH_AR[i + 1:i + p + 1])

    for i in range(iH, Ns // 2):
        HalfH_AR[i] = -np.sum(a * HalfH_AR[i - 1:i - p - 1:-1])

    H_AR = np.r_[HalfH_AR, np.conj(np.flip(HalfH_AR[1:]))]
    AR = np.fft.ifft(H_AR).real

    return AR
