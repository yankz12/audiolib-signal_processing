import numpy as np 
import scipy.signal as scsp

from dataclasses import dataclass
from scipy.fftpack import fftshift

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
    dur_fade_in : float = .01
    dur_fade_out : float = .02

    def __post_init__(self, ):
        if self.apply_fade_to:
            self._len_fade_in  = int(self.dur_fade_in*self.fs)
            self._len_fade_out = int(self.dur_fade_out*self.fs)
        else:
            self._len_fade_in  = None 
            self._len_fade_out = None

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
            s = self._apply_fade(s, where='both')

        return t, s,

    def get_hhfrfs(self, y, n_harms, len_irs = 2**12, ):
        fft_len = int(2**np.ceil(np.log2(len(y))))
        f_axis = np.linspace(0, self.fs/2, num=round(fft_len/2)+1) # frequency axis
        Y = np.fft.rfft(y, fft_len)/self.fs

        # definition of the inferse filter in spectral domain 
        # (Novak et al., "Synchronized swept-sine: Theory, application, and implementation." 
        # Journal of the Audio Engineering Society 63.10 (2015): 786-798. Eq.(43)):
        SI = 2*np.sqrt(f_axis/self._L)*np.exp(
            -1j*2*np.pi*f_axis*self._L*(1-np.log(f_axis/self.f1)) + 1j*np.pi/4
        )
        SI[0] = 0j
        # first Nyquist zone 
        H = Y*SI

        # ifft
        h = np.fft.irfft(H)

        dt = self._L*np.log(np.arange(1,n_harms + 1))*self.fs  # positions of higher orders up to N
        dt_rem = dt - np.around(dt) # The time lags may be non-integer in samples, the non integer delay must be applied later
        shft = round(len_irs/2)          # number of samples to make an artificail delay
        h_pos = np.hstack((h, h[0:shft + len_irs - 1]))  # periodic impulse response
        # separation of higher orders 
        hs = np.zeros((n_harms, len_irs))
        t_hs = np.arange(
            0,
            np.round(len_irs - 1)/self.fs,
            1/self.fs,
        )  # time axis
        axe_w = np.linspace(0, np.pi, num=int(len_irs/2+1)); # frequency axis 

        for k in range(n_harms):
            hs[k,:] = h_pos[
                len(h) - int(round(dt[k])) - shft-1:len(h) - int(round(dt[k]))
                - shft + len_irs - 1
            ]
            H_temp = np.fft.rfft(hs[k,:])

            # Non integer delay application
            H_temp = H_temp * np.exp(-1j*dt_rem[k]*axe_w)
            hs[k,:] = np.fft.irfft(H_temp)

        # Higher Harmonics
        freq_Hs = axe_w/(np.pi)*self.fs/2
        Hs = np.fft.rfft(hs)

        return t_hs, hs, freq_Hs, Hs, 

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
