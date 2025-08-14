import audiolib.plotting as al_plt
import jax.numpy as jnp
import numpy as np 
import scipy.signal as scsp

from dataclasses import dataclass, field
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
        # TODO: Make different windowing functions possible
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

@dataclass
class Multitone():
    f1 : float
    f2: float
    n_freqs : int
    fs : int
    sig_dur: float
    peak_ampl : float = 1
    ampl_freq_weights : list = field(default_factory=lambda: [None])
    ampl_freqs : list = field(default_factory=lambda: [None])
    win_type : str = 'rect'
    win_dur : float = 5e-2 # 50ms
    """
    Multitone with logarithmically spaced frequencies

    Parameters
    ----------
        f1 : float
            Lower end frequency
        f2 : float
            Higher end frequency
        n_freqs : int
            Total number of summed sine waves
        fs : int
            Sampling frequency
        sig_dur : float
            Total multitone duration in seconds
        peak_ampl : float, option
            Peak amplitude of multitone, defaults to 1
        ampl_freq_weights : array, optional
            Weighting function for individual sine waves, defaults to [None]
        ampl_freqs : array, optional
            Frequency vector according to ampl_freq_weights, defaults to [None].
            If sine frequency doesnt hit weighting frequency perfectly,
            instance chooses the closest frequency weight for this sine.
        win_type : str, optional
            Windowing function for multitone, defaults to 'rect' (= no window).
            Has to be of scipy window type
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html
        win_dur : float, optional
            Duration of applied window. Len of each fade (in/out) will be
            win_dur/2.

    Call multitone.get_multitone_signal() to get [t, signal]
    Call multitone.frequencies to get frequencies in multitone
    """
    def __post_init__(self):
        if (self.ampl_freq_weights[0] is not None) and (self.ampl_freqs[0] is not None):
            self._ampl_weight = True
            print('  Multitone: applying frequency weights.')
        else:
            self._ampl_weight = False

    def get_multitone_signal(self):
        multitone_sig = np.zeros(len(self.t))
        for f in self.frequencies:
            phase = np.pi/2 *  np.random.randn(1)
            f_weight = 1

            if self._ampl_weight:
                f_weight_idx = np.argmin(np.abs(self.ampl_freqs - f))
                f_weight = self.ampl_freq_weights[f_weight_idx]

            multitone_sig += f_weight*np.sin(2*np.pi*f * self.t + phase)

        multitone_sig = self.peak_ampl*multitone_sig/max(multitone_sig)
        # import pdb
        # pdb.set_trace()
        multitone_sig = apply_window(
            multitone_sig,
            win_type=self.win_type,
            win_dur=self.win_dur,
            fs=self.fs,
        ) if self.win_type != 'rect' else multitone_sig
        return self.t, multitone_sig

    @property
    def frequencies(self):
        freqs = np.unique(
            np.round(
                np.logspace(
                    np.log10(self.f1),
                    np.log10(self.f2),
                    self.n_freqs,
                    endpoint=True,
        )))
        return freqs
    
    @property
    def t(self):
        return np.arange(0, self.sig_dur, 1/self.fs)
    
    


def get_crest_factor(signal):
    return np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2))

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
    if isinstance(sig, jnp.ndarray):
        sig = sig.at[:int(win_len/2)].set(sig[:int(win_len/2)]*fade_in)
        sig = sig.at[-int(win_len/2):].set(sig[-int(win_len/2):]*fade_out)
    else:
        sig[:int(win_len/2)] *= fade_in
        sig[-int(win_len/2):] *= fade_out
    return sig

def _prep_harmonics_analysis(
    x,
    fs,
    f_sine,
    nfft = None,
    apply_win = False,
    win_dur = 0.01,
    num_harms = 5,
    tol_hz = 10,
):
    # ------------------------------------------------------------------------
    # Window application
    if apply_win:
        window = 'blackman'
        x = apply_window(x, fs=fs, win_type=window, win_dur=win_dur, )

    # ------------------------------------------------------------------------
    # Spectrum calculation and gathering of first frequency infos
    freq, spec = get_rfft_spec(x, fs, Nfft=nfft, )

    # ------------------------------------------------------------------------
    # Check if f_sine is perfectly hit or not in spectrum (prevent leakage)
    if not f_sine in freq:
        closest_freq = np.argmin(np.abs(freq - freq_under_study))
        raise ValueError(
            f"Harmonic Analysis: Input frequency {f_sine} Hz does not hit "
            f"frequency bin for proper harmonic amplitude estimation. Next "
            f"bin is {closest_freq} Hz. " + 
            f"Re-define your frequency or signal length or nfft."
        )
    else:
         print(f"THD: FUT {f_sine} Hz hits frequency bin perfectly.")

    freq_under_study_idx = np.argwhere(freq == f_sine)[0][0]
    freq_under_study = freq[freq_under_study_idx]
    freq_accuracy = freq[1] - freq[0]

    # ------------------------------------------------------------------------
    # Integrity checks of inputs (tolerance, Nyquist etc.)
    if tol_hz >= f_sine:
        while tol_hz >= f_sine:
            tol_hz /= 2
        print(
            'Frequency tolerance bigger than base frequency: '
            'Danger of including neighboring harmonics in tolerane range. '
            f'\nReduced tolerance to {tol_hz}Hz.')
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
            'THD calculation: Frequency tolerance is 0 Hz, evaluate ' +
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
    
    return freq, spec, freq_under_study_idx, tol_hz


def hnr_from_time_sig(
    x,
    noise,
    fs,
    f_sine,
    nfft = None,
    apply_win = False,
    win_dur = 0.01,
    num_harms = 5,
    tol_hz = 5,
    crest_lim = 1.5,
    plot_spec = True,
):
    freq, spec, freq_under_study_idx, tol_hz = _prep_harmonics_analysis(
        x = x,
        fs = fs,
        f_sine = f_sine,
        nfft = nfft,
        apply_win = apply_win,
        win_dur = win_dur,
        num_harms = num_harms,
        tol_hz = tol_hz,
    )
    freq_under_study = freq[freq_under_study_idx]
    _, noise_spec = get_rfft_spec(noise, fs, Nfft=nfft, )

    # ------------------------------------------------------------------------
    # Harmonic-to-Noise calculation
    print(f'Calc THD at {freq_under_study} Hz for {num_harms} harmonics.')
    HNR = []
    bounds = []
    all_bounds = []

    for harm in np.arange(1,num_harms+1)+1:
        eval_freq = harm*freq_under_study
        eval_freq_idx = np.argmin(np.abs(freq - eval_freq))
        print(f'Harmonic {harm} @ {np.round(eval_freq, 2)} Hz')
        bounds = [eval_freq - tol_hz, eval_freq + tol_hz]
        bounds_idcs = [
            np.argmin(np.abs(freq - bounds[0])),
            np.argmin(np.abs(freq - bounds[1])),
        ]
        crest = get_crest_factor(spec[bounds_idcs[0]:bounds_idcs[1]])
        if crest < crest_lim:
            print(f'Calculate crest factor from {bounds[0]}Hz to {bounds[1]}Hz.')
            print(f'Crest @ {eval_freq}Hz is too low ({np.round(crest,2)}). '
                  f'Should be >= {crest_lim} to assume that the harmonic is not '
                   'drowning in noise. Skipping.')
            HNR.append(0)
            continue
        print(f'Averaging noise level between {bounds[0]}Hz and {bounds[1]}Hz.')
        num = max(spec[bounds_idcs[0]:bounds_idcs[1]])
        denum = np.mean(abs(noise_spec[bounds_idcs[0]:bounds_idcs[1]]))
        HNR.append(num / denum)
        all_bounds.append([bounds[0], bounds[1]])

    # ------------------------------------------------------------------------
    # Plotting
    if plot_spec:
        vals = 20*np.log10(abs(spec))
        noise_spec_log = 20*np.log10(abs(noise_spec))
        fig, ax = al_plt.plot_rfft_freq(
            f = freq,
            data = vals,
            xscale = 'lin',
        )
        al_plt.plot_rfft_freq(
            f = freq,
            data = noise_spec_log,
            xscale = 'lin',
            fig=fig,
            ax=ax,
        )
        ax.set(
            title='THD Spectrum',
            xlim = [freq_under_study - 10, eval_freq + 10],
            ylim = [
                min(vals[freq_under_study_idx:eval_freq_idx]) - 10,
                max(vals[freq_under_study_idx:eval_freq_idx]) + 10,
            ]
        )
        
        if tol_hz != 0:
            ylims = ax.get_ylim()
            _ = [
                ax.fill_betweenx(
                    ylims, bound[0], bound[1], alpha=.35
                ) for bound in all_bounds
            ]

    return HNR

def thd_from_time_sig(
        x,
        fs,
        f_sine,
        nfft = None,
        apply_win = False,
        win_dur = 0.01,
        num_harms = 5,
        tol_hz = 10,
        plot_spec = False,
    ):
    """ 
    Calculate THD from pure sine time signal using blackman-harris window.
    Input sine frequency needs to be known beforehand.

    THD = √(Y(f_sine*2)^2 + Y(f_sine*3)^2 + ... + Y(f_sine*n)^2))
          -------------------------------------------------------
                            Y(f_sine)

    Parameters
    ----------
    x : iterable (e.g. np.ndarray)
        Time signal to calculate THD from
    fs : int
        Sampling frequency of x
    f_sine : float
        Frequency of input sine
    nfft : int
        Number of FFT Points. If nfft > len(x), signal gets zero-padded.
        If nfft < len(x), then the input is cropped.
    apply_win : bool, defaults to False
        Wether to apply Blackman-Harris window function to x. 
        Blackman-Harris is usually the best window for THD measurements.
    win_dur : float, optional, defaults to 0.01 [s] (10 [ms])
        Length of the window. Len of each fade (in/out) will be win_dur/2.
    num_harms : int, optional, defaults to 5
        Number of harmonics to include in THD calculation. Is automatically
        limited if n*freq_under_study > fs/2. Fundamental = 0th harmonic
        --> e.g. if num_harms == 5, maximum frequency to be included will
            be 5*f_sine
    tol_hz : float, defaults to 10 (= 10Hz)
        Tolerance in Hertz:
        Include frequency bins around analyzed frequencies in order to get
        leaked energy in the specturm into the THD value. E.g. when analyzing
        f1, the calc will include all bins from f1 - tolerance to
        f1 + tolerance
    plot_spec : boolean, defaults to False
        Plot spectrum of windowed signal or not within harmonic freq. range

    Returns
    -------
    thd : float
        Total Harmonic distortion in percentage
    """

    freq, spec, freq_under_study_idx, tol_hz = _prep_harmonics_analysis(
        x = x,
        fs = fs,
        f_sine = f_sine,
        nfft = nfft,
        apply_win = apply_win,
        win_dur = win_dur,
        num_harms = num_harms,
        tol_hz = tol_hz,
    )
    freq_under_study = freq[freq_under_study_idx]

    # ------------------------------------------------------------------------
    # THD calculation
    print(f'Calc THD at {freq_under_study} Hz for {num_harms} harmonics.')
    thd_num = 0
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

    # ------------------------------------------------------------------------
    # Plotting
    if plot_spec:
        vals = 20*np.log10(abs(spec))
        _, ax = al_plt.plot_rfft_freq(
            f = freq,
            data = vals,
            xscale = 'lin',
        )
        ax.set(
            title='THD Spectrum',
            xlim = [freq_under_study - 10, eval_freq + 10],
            ylim = [
                min(vals[freq_under_study_idx:eval_freq_idx]) - 10,
                max(vals[freq_under_study_idx:eval_freq_idx]) + 10,
            ]
        )
        
        if tol_hz != 0:
            ylims = ax.get_ylim()
            _ = [
                ax.fill_betweenx(
                    ylims, bound[0], bound[1], alpha=.35
                ) for bound in all_bounds
            ]

    return 100 * thd_num / thd_denum

def get_group_delay(freq, H, ):
    """
    Calculates group delay spectrum of complex-valued transfer fuction H.
    Note that this calculation does not include sample-based delay (linear
    phase delay), as it is always normalized to 0s. To get sample

    Parameters
    ----------
    freq : array or iterable [Hz]
        Frequency vector of H
    H : array or iterable
        Complex-value transfer function of the system under study
    
    Returns
    -------
    group_delay : np.ndarray [s]
        Group delay spectrum of H
    """
    phase_spec = np.unwrap(np.angle(H))
    w = 2*np.pi*freq
    group_delay = -jnp.gradient(phase_spec, w)
    return group_delay

def get_delay_via_crosscorr(x, y, fs, plot=False):
    cross_corr = scsp.correlate(x, y, mode='full', method='direct')
    lags = scsp.correlation_lags(x.size, y.size, mode='full')/fs
    # Smiths suggestion to improve guess at x-corr peak, when
    # the "correct" delay lies between two sample points:
    # https://ccrma.stanford.edu/%7Ejos/parshl/Peak_Detection_Steps_3.html
    x_corr_max_idx = np.argmax(cross_corr)
    # alpha = lags[x_corr_max_idx-1]
    beta = lags[x_corr_max_idx]
    # gamma = lags[x_corr_max_idx+1]
    # p_num = alpha - gamma
    # p_denum = alpha - 2*beta + gamma
    # print(f'alpha : {alpha}, beta : {beta}, gamma : {gamma}')
    # print(f'Numerator: {p_num}, Denum: {p_denum}')
    # if p_denum == 0:
    #     print(f'Denum is 0')
    #     delay = beta
    # else:
    #     delay = beta + .5*p_num/p_denum
    if plot:
        fig, ax = al_plt.plot_time(lags*1e3, cross_corr)
        ax.set(xlabel='Lag [ms]')
    return beta

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
