import audiolib.plotting as al_plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc_sp

from audiolib.signal_processing import ExpSweep
from SynchSweptSine import SynchSweptSine


"""
Test Implementation fo Novaks Exponential Sweep by
    1. calculate b-coeffs of FIR low-pass @ fs/4 cutoff
    2. Create sweep signal starting at 2 kHz
    2. convolute filter with sweep signal
    3. extract HHFRFs 
"""


# plt.close('all')

# ----------------------------------------------------------------------------
# Sweep definitions
fs = 48000
f1 = 1e3
f2 = 8e3
dur = 1
apply_fade_to = 'both'
len_irs = 2**13
n_harms = 3



# ----------------------------------------------------------------------------
# Sweep creation
sweep = ExpSweep(
    fs=fs,
    f1=f1,
    f2=f2,
    approx_dur = dur,
)
t_sweep, s_sweep = sweep.get_sweep_signal()
sss = SynchSweptSine(f1=f1, f2=f2, T=dur, fs=fs, fade=[0, 0])
sss_sweep = sss.signal

# ----------------------------------------------------------------------------
# Convolution of sweep and linear test filter
y = s_sweep + 0.025*s_sweep**2 + 0.025*s_sweep**3
y_sss = sss_sweep + 0.025*sss_sweep**2 + 0.025*sss_sweep**3

# ----------------------------------------------------------------------------
# Extraction of higher harmonic frequency functions
t_hs, hs, freq_Hs, Hs, dt = sweep.get_hhfrfs(
     y=y,
     n_harms=n_harms,
     len_irs = len_irs
)

h_sss = sss.getIR(y=y) # the full impulse response
freq_sss = np.fft.rfftfreq(len_irs, 1/fs) # Freq. axis
Hs_sss = (
    sss.separate_IR(h_sss, N=n_harms, n_samples=len_irs)
)
Hs = np.array([
    Hs_t*np.exp(-1j*freq_Hs*2*np.pi*len_irs/2/fs) for Hs_t in Hs
]) # TODO: Fix! Undo the shift of IRs by Novak in order to get proper Phase response.

Hs_sss = np.array([
    Hs_t*np.exp(-1j*freq_sss*2*np.pi*len_irs/2/fs) for Hs_t in Hs_sss
])

# ----------------------------------------------------------------------------
# Plotting higher harmonic frequency functions
fig, ax_mag, ax_arg = al_plt.plot_mag_phase(
    freq_h=freq_Hs,
    magnitude = 20*np.log10(abs(Hs.transpose())),
    phase_deg = np.angle(Hs.transpose())/np.pi*180,
    xscale='log',
)
al_plt.plot_mag_phase(
    freq_h=freq_sss,
    magnitude = 20*np.log10(abs(Hs_sss.transpose())),
    phase_deg = np.angle(Hs_sss.transpose())/np.pi*180,
    fig = fig,
    ax_mag=ax_mag,
    ax_arg=ax_arg,
)
ax_mag.axvline(
    x=f1,
    ymin=ax_mag.get_ylim()[0],
    ymax=ax_mag.get_ylim()[1],
    c='k',
)
ax_mag.axvline(
    x=fir_cutoff,
    ymin=ax_mag.get_ylim()[0],
    ymax=ax_mag.get_ylim()[1],
    c='r',
)
ax_mag.legend(
    ('1st harm', '2nd harm', '3rd harm', 'freqz', 'Sweep start freq', 'Filter Cutoff'),
    loc='center',
)
ax_mag.set_xlim(1e3, fs/2)
plt.show(block=False)
