import numpy as np 
import scipy.signal as scsp

from dataclasses import dataclass
from scipy.fftpack import fftshift

import pdb

@dataclass
class ExpSweep():
    """
    This class is copied and adjusted from Antonin Novaks Synchronized Swept
    Sine code, which is to be found in
        https://ant-novak.com/pages/sss/
    The underlying paper is 
        (Novak et al., "Synchronized swept-sine: Theory, application, and
        implementation." , Journal of the Audio Engineering Society 63.10
        (2015): 786-798.

    Parameters
    ----------
    f1 : float
        starting frequency of sweep
    f2 : float
        stop frequency of sweep
    approx_dur : float
        desired duration of sweep. Final time vector might slightly differ
        from this duration due to Novaks time adjustment of sweep (see
        underlying paper)
    apply_fade_to : str, defaults to 'both'
        Wether or not to apply fade-in and/or fade-out 
        Choose from 'in', 'out', 'both', 'None'
    dur_fade_in : float, defaults to 0.01
        Duration of fade-in window in seconds, ignored if apply_fade_to=='None'
    dur_fade_out: float, defaults to 0.02
        Duration of fade-out window in seconds, ignored if apply_fade_to=='None'
    """
    f1 : float
    f2: float
    fs : int
    approx_dur: float
    apply_fade_to : str = 'both'
    dur_fade_in : float =  .01
    dur_fade_out : float = .02

    def __post_init__(self, ):
        if self.apply_fade_to:
            self._len_fade_in  = int(self.dur_fade_in*self.fs)
            self._len_fade_out = int(self.dur_fade_out*self.fs)
            print(self.apply_fade_to)
        else:
            self._len_fade_in  = None 
            self._len_fade_out = None
            print('None')

        self._L = self.approx_dur/np.log(self.f2/self.f1)

    def get_sweep_signal(self, ):
        """
        Gives the time signal of the exponential sweep (= input signal for
        measurement).
        
        Returns
        -------
        t : np.ndarray
            Time vector of sweep
        s : np.ndarray
            Sweep
        """
        t = np.arange(
            0,
            np.round(self.fs * self.approx_dur - 1)/self.fs,
            1/self.fs,
        )  # time axis
        s = np.sin(2*np.pi*self.f1*self._L*np.exp(t/self._L)) # generated swept-sine signal

        if self.apply_fade_to:
            s = self._apply_fade(s, where=self.apply_fade_to)

        return t, s,

    def Xinv(self, Npts):
        ''' calculates Xinv = 1/X, where X is the Fourier Transform of the swept-sine  '''
        import warnings
        warnings.filterwarnings("ignore")
        # suppress warnings temporarily (log of zero in Xinv definition)

        # definition of the inferse filter in spectral domain
        # (Novak et al., "Synchronized swept-sine: Theory, application, and implementation."
        # Journal of the Audio Engineering Society 63.10 (2015): 786-798.
        # Eq.(43))
        f_axis = self.f_axis(Npts)
        Xinv = 2*np.sqrt(f_axis/self._L)*np.exp(-1j*2*np.pi *
            f_axis*self._L*(1-np.log(f_axis/self.f1)) + 1j*np.pi/4)
        Xinv[0] = 0j

        warnings.filterwarnings("default")
        return Xinv

    def getIR(self, y, ):
        # FFT of the output signal
        Y = np.fft.rfft(y)/self.fs
        # complete FRF
        H = Y*self.Xinv(len(y))
        return np.fft.irfft(H)

    def f_axis(self, Npts):
        return np.fft.rfftfreq(Npts, d=1.0/self.fs)

    def separate_IR(self, h, N=3, n_samples=2**12, latency=0):
        ''' Separates the nonlinear contributions in the impulse response h
            and calculates their Fourier Transform to get the Higher Harmonic
            Frequency Responses (HHFRs).'''
        dt = self._L*np.log(np.arange(1, N+1)) * \
            self.fs  # positions of higher orders up to N
        # The time lags may be non-integer in samples, the non integer delay must be applied later
        dt_rem = dt - np.around(dt)

        # number of samples to make an artificail delay
        shft = int(n_samples/2)
        # periodic impulse response
        h_pos = np.concatenate(
            (h[latency:], h[0:shft + latency + n_samples - 1]))

        # separation of higher orders
        hs = np.zeros((N, n_samples))

        w_normalized = np.fft.rfftfreq(n_samples, d=1.0/(2*np.pi))
        for k in range(N):
            st  = len(h) - int(round(dt[k])) - shft - 1
            end = st + n_samples
            hs[k, :] = h_pos[st:end]
            H_temp = np.fft.rfft(hs[k, :])

            # Non integer delay application
            H_temp = H_temp * np.exp(-1j*dt_rem[k]*w_normalized)
            hs[k, :] = np.fft.irfft(H_temp)

        # Higher Harmonics
        freq = self.f_axis(len(hs[0]))
        Hs = np.fft.rfft(hs)
        return freq, Hs, hs, dt, 

    def get_hhfrfs(self, y, n_harms, len_irs = 2**12, ):
        hs = self.getIR(y) # the full impulse response
        freq, Hs, hs, dt = self.separate_IR(hs, N=n_harms, n_samples=len_irs)    # separatef HHFRs
        t = np.arange(0, np.round(len(hs[0]))/self.fs,1/self.fs)  # time axis
        # Hs = Hs*np.exp(-1j*freq*2*np.pi*len_irs/2/self.fs)

        return t, hs, freq, Hs, dt,

    def _apply_fade(self, sweep, where, ):
        """
        Parameters
        ----------
        sweep : np.ndarray
            sweep signal to be faded in and/or out
        where : str, one out of ['in', 'out', 'both']
            Apply only fade-in, only fade-out or both
        """
        if where == 'in' or where == 'both':
            sweep[0:self._len_fade_in] = sweep[0:self._len_fade_in] * (
                (
                    -np.cos(np.arange(self._len_fade_in)
                    / self._len_fade_in*np.pi)+1
                ) / 2
            )
        if where == 'out' or where == 'both':
            sweep[-self._len_fade_out:] = sweep[-self._len_fade_out:] *  (
                (
                    np.cos(np.arange(self._len_fade_out)
                    / self._len_fade_out*np.pi)+1
                ) / 2
            )
        return sweep
    
    @property
    def final_dur(self):
        # Formula (32) in Novak, 2015
        k = int(np.round(self.f1*self.approx_dur/(np.log(self.f2/self.f1))))
        T = k*np.log(self.f2/self.f1)/self.f1
        return T


def get_rfft_spec(x, fs, Nfft=None):
    if Nfft is None:
        Nfft = len(x)
    freq = np.fft.rfftfreq(Nfft, 1/fs)
    spec = np.abs(np.fft.rfft(x, Nfft))
    return freq, spec

def get_rfft_power_spec(x, fs, Nfft=None):
    freq, spec = get_rfft_spec(x, fs, Nfft)
    Sxx = spec**2
    return freq, Sxx

def get_ir_from_rfft(spec, fs, Nfft):
    """
    Computes real-valued IR from spectrum. Spectrum is expected to be computed
    with np.fft.rfft

    Parameters
    ----------
    spec : array of signal
        Spectrum of signal, computed with np.fft.rfft
    fs : int
        Sampling frequency of x
    Nfft : int
        Number of FFT and IFFT points. Corresponds to the number of points of the 
        resulting IR. If Nfft < len(x), x is zero-padded.

    Returns
    -------
    t : numpy array
        Time vector of resulting IR, centered at t = 0 in the middle of array
    centered_ir : numpy array
        Impulse Response
    """
    centered_ir = Nfft * fftshift(np.real(np.fft.irfft(spec, n=Nfft)))
    t = np.arange(-int(Nfft/2),int(Nfft/2)) / fs
    return t, centered_ir

def get_ir_from_rawdata(x, fs, Nfft):
    """
    Computes real-valued IR from rawdata set.

    Parameters
    ----------
    x : array of signal
        Input signal, assumed to be completely real
    fs : int
        Sampling frequency of x
    Nfft : int
        Number of FFT and IFFT points. Corresponds to the number of points of the 
        resulting IR. If Nfft < len(x), x is zero-padded.

    Returns
    -------
    t : numpy array
        Time vector of resulting IR, centered at t = 0 in the middle of array
    centered_ir : numpy array
        Impulse Response
    """
    _, spec = get_rfft_spec(x, fs, Nfft)
    t, centered_ir = get_ir_from_rfft(spec, fs, Nfft)
    return t, centered_ir

def get_msc(sig_0, sig_1, fs, blocklen, ):
    # TODO: Test
    print('Not tested, use with caution!')
    freq_msc, msc = scsp.coherence(
        sig_0,
        sig_1,
        fs,
        nperseg=blocklen,
    )
    return freq_msc, msc
