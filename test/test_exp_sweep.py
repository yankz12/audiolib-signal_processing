import audiolib.plotting as al_plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc_sp

from audiolib.signal_processing import ExpSweep

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
sig_ampl = 1
dur = 1
apply_fade_to = 'both'
len_irs = 2**12
n_harms = 3


# ----------------------------------------------------------------------------
# Sweep creation
sweep = ExpSweep(
    fs=fs,
    f1=f1,
    f2=f2,
    ampl=sig_ampl,
    approx_dur = dur,
    num_harmonics=n_harms,
    len_irs = len_irs,
)
t_sweep, s_sweep = sweep.get_sweep_signal()

# ----------------------------------------------------------------------------
# Non-Linear System
y = s_sweep + 0.25*s_sweep**2 + 0.25*s_sweep**3

# ----------------------------------------------------------------------------
# Extraction of higher harmonic frequency functions
t_hs, hs, freq_Hs, Hs, dt = sweep.get_hhfrfs(y=y)
Hs = sweep.revert_delay(Hs)

# ----------------------------------------------------------------------------
# Calc THD
freq_thd, thd = sweep.get_thd(y=y)

# ----------------------------------------------------------------------------
# Frequency limits of harmonics
first_harm_lim = [f1, f2]
sec_harm_lim = [2*f1, 2*f2]
third_harm_lim = [3*f1, 3*f2]
first_harm_lim_idcs = [np.argmin(np.abs(freq_Hs - f)) for f in first_harm_lim]
sec_harm_lim_idcs = [np.argmin(np.abs(freq_Hs - f)) for f in sec_harm_lim]
third_harm_lim_idcs = [np.argmin(np.abs(freq_Hs - f)) for f in third_harm_lim]
lim_idcs = [first_harm_lim_idcs, sec_harm_lim_idcs, third_harm_lim_idcs, ]

# ----------------------------------------------------------------------------
# Plotting higher harmonic frequency functions
fig, ax_mag, ax_arg = al_plt.plot_mag_phase(
    f=freq_Hs,
    magnitude = 20*np.log10(abs(Hs.transpose())),
    phase_deg = np.angle(Hs.transpose())/np.pi*180,
    phase_xlim_idcs = lim_idcs,
    xscale = 'log',
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

# ----------------------------------------------------------------------------
# Plotting THD
fig, ax = al_plt.plot_rfft_freq(
    f = freq_thd,
    data = thd,
    xscale='log',
)
ax.set(ylabel = 'THD [%]')

plt.show(block=False)
