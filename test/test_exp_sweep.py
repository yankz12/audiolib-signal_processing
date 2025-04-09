import audiolib.plotting as al_plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc_sp

from audiolib.signal_processing import ExpSweep
from SynchSweptSine import SynchSweptSine


"""
Test Implementation fo Novaks Exponential Sweep by
    1. Create sweep signal sig
    2. Apply non-linear system by
        y = sig + 0.025*sig**2 + 0.015*sig**3
    3. extract and plot HHFRFs 
"""

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
    num_harmonics=n_harms,
    len_irs = len_irs,
)
t_sweep, s_sweep = sweep.get_sweep_signal()
sss = SynchSweptSine(f1=f1, f2=f2, T=dur, fs=fs, fade=[int(0.01*fs), int(0.02*fs)])
sss_sweep = sss.signal

# ----------------------------------------------------------------------------
# Convolution of sweep and linear test filter
y = s_sweep + 0.025*s_sweep**2 + 0.025*s_sweep**3
y_sss = sss_sweep + 0.025*sss_sweep**2 + 0.025*sss_sweep**3

# ----------------------------------------------------------------------------
# Extraction of higher harmonic frequency functions
t_hs, hs, freq_Hs, Hs, dt = sweep.get_hhfrfs(
     y=y,
)
Hs = sweep.revert_delay(Hs)

# ----------------------------------------------------------------------------
# Plotting higher harmonic frequency functions
fig, ax_mag, ax_arg = al_plt.plot_mag_phase(
    freq_h=freq_Hs,
    magnitude = 20*np.log10(abs(Hs.transpose())),
    phase_deg = np.angle(Hs.transpose())/np.pi*180,
    xscale='log',
)
ax_mag.axvline(
    x=f1,
    ymin=ax_mag.get_ylim()[0],
    ymax=ax_mag.get_ylim()[1],
    c='k',
)
ax_mag.legend(
    ('1st harm', '2nd harm', '3rd harm', 'Sweep start freq',),
    loc='center',
)
ax_mag.set(
    xlim=(500, fs/2),
    title=r'$y = x + 0.25x^2 + 0.25x^3$'
)
ax_mag.set_xlim(500, fs/2)

plt.show(block=False)
