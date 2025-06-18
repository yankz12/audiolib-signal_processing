import audiolib.plotting as al_plt
import jax.numpy
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
    num_harmonics : int, defaults to 3
        Number of HHFRFs to be extracted
    len_irs : int, defaults to 2**12
        Length of the returned impulse responses of the HHFRFs
    
    """
    f1 : float
    f2: float
    fs : int
    approx_dur: float
    ampl : float = 1
    apply_fade_to : str = 'both'
    dur_fade_in : float =  .01
    dur_fade_out : float = .02
    num_harmonics : int = 3
    len_irs : int = 2**12

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
        s = self.ampl*np.sin(2*np.pi*self.f1*self._L*np.exp(t/self._L)) # generated swept-sine signal

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
        Xinv = self.ampl*2*np.sqrt(f_axis/self._L)*np.exp(-1j*2*np.pi *
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

    def separate_IR(self, h, latency=0):
        ''' Separates the nonlinear contributions in the impulse response h
            and calculates their Fourier Transform to get the Higher Harmonic
            Frequency Responses (HHFRs).'''
        dt = self._L*np.log(np.arange(1, self.num_harmonics+1)) * \
            self.fs  # positions of higher orders up to N
        # The time lags may be non-integer in samples, the non integer delay must be applied later
        dt_rem = dt - np.around(dt)

        # number of samples to make an artificail delay
        shft = int(self.len_irs/2)
        # periodic impulse response
        h_pos = np.concatenate(
            (h[latency:], h[0:shft + latency + self.len_irs - 1]))

        # separation of higher orders
        hs = np.zeros((self.num_harmonics, self.len_irs))

        w_normalized = np.fft.rfftfreq(self.len_irs, d=1.0/(2*np.pi))
        for k in range(self.num_harmonics):
            st  = len(h) - int(round(dt[k])) - shft - 1
            end = st + self.len_irs
            hs[k, :] = h_pos[st:end]
            H_temp = np.fft.rfft(hs[k, :])

            # Non integer delay application
            H_temp = H_temp * np.exp(-1j*dt_rem[k]*w_normalized)
            hs[k, :] = np.fft.irfft(H_temp)

        # Higher Harmonics
        freq = self.f_axis(len(hs[0]))
        Hs = np.fft.rfft(hs)
        return freq, Hs, hs, dt, 

    def get_hhfrfs(self, y, ):
        """
        Parameters
        ----------
        y : np.ndarray
            Output signal of system under study
        
        Returns
        -------
        t : np.ndarray
            Time vector of impulse responses
        hs : np.ndarray
            Matrix of higher harmonic impulse responses (HHIRs)
        freq : np.ndarray
            Frequency vector of higher harmonic frequency responses (HHFRFs)
        Hs : np.ndarray
            Higher harmonic frequency responses (HHFRFs)
        dt : list
            List of time delays applied to each impulse response in order to
            be synchronized (Novak, 2015)
        """
        hs = self.getIR(y) # the full impulse response
        freq, Hs, hs, dt = self.separate_IR(hs)    # separatef HHFRs
        t = np.arange(0, np.round(len(hs[0]))/self.fs,1/self.fs)  # time axis
        # Hs = Hs*np.exp(-1j*freq*2*np.pi*len_irs/2/self.fs)

        return t, hs, freq, Hs, dt,

    def get_thd(self, y, ):
        _, _, freq, Hs, _, = self.get_hhfrfs(y)
        idx_f1 = np.argmax(freq >= self.f1)  # Find the starting index for f1
        idx_f2 = np.argmax(freq >  self.f2)  # Find the ending index for f2
        f_indexes = np.arange(idx_f1, idx_f2)  # f_indexes for the range f1 to f2
        freq_thd = freq[idx_f1:idx_f2]

        # Prepare the numerator and denominator for THD calculation
        numerator = 0
        for harmonic in range(2, self.num_harmonics+1):
            numerator += np.abs(Hs[harmonic-1, harmonic*f_indexes])**2
        denumerator = np.abs(Hs[0, f_indexes])

        # Compute THD
        THD = 100 * np.sqrt(numerator) / denumerator
        return freq_thd, THD

    def revert_delay(self, Hs, ):
        """
        Reverts the delay of len_irs/2 in frequency domain in order to have
        correct phase response of the system under study. get_hhfrfs()
        returns the IRs centered in the applied window of len_irs for
        post-processing of the IRs e.g. convolution with other signals.
        However, this falsifies the phase response of the system under study.
        Therefore, the delay should be reverted in case that the system is
        to be studied instead of being post-processed.

        Hs_reverted = Hs * e^(-jw*phi),
        with phi being len_irs/2

        Parameters
        ----------
        Hs : np.ndarray
            matrix containing the delayed complex-valued FRFs of all harmonics

        Returns
        -------
        Hs_reverted : np.ndarray
            Matrix with HHFRFs without delay of len_irs/2
        """
        # Hs_reverted = np.array([
        #     Hs_t*np.exp(-1j*2*np.pi*self.f_axis(self.len_irs)/self.fs*(self.len_irs/2)
        #     )
        #     for Hs_t in Hs
        # ])
        Hs_reverted = Hs*np.exp(
            -1j*2*np.pi*self.f_axis(self.len_irs)/self.fs*3*(self.len_irs)/2
        )
        return Hs_reverted

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


def _find_two_closest_bins(f, f_axis, ):
    """
    Findet die zwei nächsten Frequenz-Bins zu einer gegebenen Frequenz im Spektrum.

    :param frequenz: Die Ziel-Frequenz, für die die Bins gesucht werden.
    :param spektrum_binning: Ein Array oder eine Liste mit den Frequenz-Bins des Spektrums.
    :return: Ein Tuple mit den Indizes der zwei nächsten Bins.
    """
    # Berechne die Differenzen zwischen der Ziel-Frequenz und jedem Bin
    diffs = [abs(f_bin - f) for f_bin in f_axis]
    # Sortiere die Indizes nach der Differenz
    sorted_idcs = sorted(range(len(diffs)), key=lambda i: diffs[i])
    # Die zwei nächsten Bins sind die ersten beiden in der sortierten Liste
    return sorted_idcs[0], sorted_idcs[1]

def apply_window(sig, fs, win_type, win_dur, ):
    """
    Assumes symmetrical window. Len of each fade (in/out) will be win_dur/2.

    Parameters
    ----------
    sig : array
        Signal to be windowed
    fs : int
        Sampling frequency
    win_type : str
        Type of window that will be applied. Has to be of scipy window type
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
    win_dur : float
        Duration of applied window. Len of each fade (in/out) will be win_dur/2.

    Returns 
    -------
    win_sig : array
        Windowed signal
    """
    win_len = int(np.round(win_dur*fs))
    win_len_is_uneven = win_len % 2
    if win_len_is_uneven:
        win_len += 1 # To ensure a 1 at maximum of window
    win = scsp.get_window(window=win_type, Nx=win_len)
    win_max = np.argmax(win)
    fade_in = win[:win_max]
    fade_out = np.flip(fade_in)
    # Apply window:
    if isinstance(sig, jax.numpy.ndarray):
        sig = sig.at[:int(win_len/2)].set(sig[:int(win_len/2)]*fade_in)
        sig = sig.at[-int(win_len/2):].set(sig[-int(win_len/2):]*fade_out)
    else:
        sig[:int(win_len/2)] *= fade_in
        sig[-int(win_len/2):] *= fade_out
    return sig


def thd_from_time_sig(
        x,
        fs,
        f_sine,
        apply_win = False,
        win_dur = 0.01,
        num_harms = 5,
        tol_hz = 10,
        plot_spec = False,
    ):
    """ 
    Calculate THD from time signal using blackman-harris window. Always
    chooses frequency of maximum amplitude as f0.

    THD = √(Y(f0*2)^2 + Y(f0*3)^2 + ... + Y(f0*n)^2))
            ----------------------------------------
                            Y(f0)

    Parameters
    ----------
    x : iterable (e.g. np.ndarray)
        Time signal to calculate THD from
    fs : int
        Sampling frequency of x
    apply_win : bool, defaults to False
        Wether to apply Blackman-Harris window function to x. 
        Blackman-Harris is usually the best window for THD measurements.
    win_dur : float, optional, defaults to 0.01 [s] (10 [ms])
        Length of the flat-top window
    num_harms : int, optional, defaults to 5
        Number of harmonics to include in THD calculation. Is automatically
        limited if n*freq_under_study > fs/2. Fundamental = zeroth harmonic
        --> e.g. if num_harms == 5, maximum frequency to be included will
            be 5*
    tol_perc : float, defaults to 10 (= 10 %)
        Tolerance percentage:
        Include frequency bins around analyzed frequencies in order to get
        leaked energy in the specturm into the THD value. E.g. when analyzing
        f1, the calc will include all bins from f1 - tolerance to
        f1 + tolerance

    Returns
    -------
    thd : float
        Total Harmonic distortion in percentage
    """
        # Check if the target frequency is in the frequency bins
    
    if apply_win:
        window = 'blackman'
        x = apply_window(x, fs=fs, win_type=window, win_dur=win_dur, )

    freq, spec = get_rfft_spec(x, fs, )
    freq_under_study_idx = np.argmax(abs(spec))
    freq_under_study = freq[freq_under_study_idx]
    freq_accuracy = freq[1] - freq[0]

    if f_sine in freq:
        print(f"Frequency {f_sine} Hz hits frequency bin perfectly.")
    else:
        raise ValueError(
            f"Input frequency {f_sine} Hz does not hit frequency bin for " +
            "proper THD amplitude estimation. " + 
            f"Re-define your frequency or signal length."
        )

    if (tol_hz < freq_accuracy) and (tol_hz != 0):
        print(79*'-')
        print(
            ' THD Calculation:\n Tolerance too small: ' + 
            f'set to minimum possible tolerance of ' + 
            f'{np.round(freq_accuracy, 2)} Hz.\n (= frequency resolution)'
        )
        print(79*'-')
        tol_hz = freq_accuracy
    if tol_hz == 0:
        print(
            'THD calculation: Frequency tolerance is zero, evaluating ' +
            'single frequency bins only.'
        )
        print(79*'-')

    if num_harms*freq_under_study > fs/2:
        num_harms = int(fs/freq_under_study/2)
        print(79*'-')
        print(
            f'THD calc: Limiting number of harmonics to {num_harms} ' + 
            'to stay within Nyquist range.'
        )
        print(79*'-')

    print(f'Calc THD at {freq_under_study} Hz for {num_harms} harmonics.')

    thd_num = 0
    eval_freqs = []
    bounds = []
    all_bounds = []

    for harm in np.arange(1,num_harms+1)+1:
        eval_freq = harm*freq_under_study
        eval_freq_idx = np.argmin(np.abs(freq - eval_freq))
        print(f'Harmonic {harm} @ {np.round(eval_freq, 2)} Hz')
        if tol_hz == 0:
            thd_num += abs(spec[eval_freq_idx])**2
            continue
        bounds = [eval_freq - tol_hz, eval_freq + tol_hz]
        bounds_idcs = [
            np.argmin(np.abs(freq - bounds[0])),
            np.argmin(np.abs(freq - bounds[1])),
        ]
        print(f'Summing from {bounds[0]} to {bounds[1]}')
        thd_num += sum(abs(spec[bounds_idcs[0]:bounds_idcs[1]]))**2
        all_bounds.append([bounds[0], bounds[1]])

    thd_num = np.sqrt(thd_num)
    if tol_hz == 0:
        thd_denum = abs(spec[freq_under_study_idx])
    else:
        low_bound =  freq_under_study - tol_hz
        high_bound = freq_under_study + tol_hz
        low_bound_idx = np.argmin(np.abs(freq - low_bound))
        high_bound_idx = np.argmin(np.abs(freq - high_bound))
        bounds.append([freq[low_bound_idx], freq[high_bound_idx]])
        thd_denum = sum(abs(spec[low_bound_idx:high_bound_idx]))

    if plot_spec:
        fig, ax = al_plt.plot_rfft_freq(
            f = freq,
            data = 20*np.log10(abs(spec)),
            xscale = 'lin',
        )
        ax.set(
            title='THD Spectrum',
            xlim = [freq_under_study - 10, eval_freq + 10]
        )
        
        if tol_hz != 0:
            ylims = ax.get_ylim()
            _ = [
                ax.fill_betweenx(
                    ylims, bound[0], bound[1], alpha=.35
                ) for bound in all_bounds
            ]

    return 100 * thd_num / thd_denum



    

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
