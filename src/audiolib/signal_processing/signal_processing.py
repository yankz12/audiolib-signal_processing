import numpy as np 
import scipy.signal as scsp

from scipy.fftpack import fftshift
from typing import Optional

class NovakSweep():
    f1 : float
    f2: float
    fs : int
    approx_dur: float
    apply_fade : bool
    dur_fade_in : Optional[float]
    dur_fade_out : Optional[float]

    def __post_init__(self, ):
        if self.apply_fade:
            self._samples_fade_in  = int(self.fade_len_in*self.fs)
            self._samples_fade_out = int(self.fade_len_out*self.fs)
        else:
            self._samples_fade_in  = None 
            self._samples_fade_out = None

        self._L = self.approx_dur/np.log(self.f2/self.f1)

    def get_sweep_signal(self, ):
        t = np.arange(0,np.round(self.fs*self.approx_dur-1)/self.fs,1/self.fs)  # time axis
        s = np.sin(2*np.pi*self.f1*self._L*np.exp(t/self._L))       # generated swept-sine signal

        if self.apply_fade:
            s = self._apply_fade_in_out(s, 'both')

        return t, s,

    def _apply_fade_in_out(self, sweep, where, ):
        """
        Parameters
        ----------
        sweep : np.ndarray
            sweep signal to be faded in and/or out
        where : str, one out of ['in', 'out', 'both']
            Apply only fade-in, only fade-out or both
        """
        if where == 'in' or where == 'both':
            sweep[0:self._samples_fade_in] = sweep[0:self._samples_fade_in] * (
                (
                    -np.cos(np.arange(self._samples_fade_in)
                    / self._samples_fade_in*np.pi)+1
                ) / 2
            )
        if where == 'out' or where == 'both':
            sweep[-self._samples_fade_out:] = sweep[-self._samples_fade_out:] *  (
                (
                    np.cos(np.arange(self._samples_fade_out)
                    / self._samples_fade_out*np.pi)+1
                ) / 2
            )
        return sweep



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
