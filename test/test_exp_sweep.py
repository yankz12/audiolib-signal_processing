import audiolib.plotting as al_plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc_sp

from audiolib.signal_processing import ExpSweep

plt.close('all')

# ----------------------------------------------------------------------------
# Sweep definitions
fs = 48000
f1 = 2e3
f2 = 20e3
dur = 5
apply_fade_to = 'both'
len_irs = 2**12
n_harms = 1

# ----------------------------------------------------------------------------
# Linear test filter definition 
fir_len = 100
fir_cutoff = fs/4
fir = sc_sp.firwin(numtaps=100, cutoff=fs/4, fs=fs,)

# ----------------------------------------------------------------------------
# Sweep creation
sweep = ExpSweep(
    fs=fs,
    f1=f1,
    f2=f2,
    approx_dur = dur,
)
t_sweep, s_sweep = sweep.get_sweep_signal()

# ----------------------------------------------------------------------------
# Convolution of sweep and linear test filter
y = sc_sp.lfilter(b = fir, a = [1, 0], x = s_sweep, )

# ----------------------------------------------------------------------------
# Extraction of higher harmonic frequency functions
t_hs, hs, freq_Hs, Hs = sweep.get_hhfrfs(
     y=y,
     n_harms=n_harms,
     len_irs = len_irs
)
Hs_correct_phase = [Hs_tmp*np.exp(-1j*freq_Hs*np.pi*len_irs/fs) for Hs_tmp in Hs]
Hs_correct_phase = np.array(Hs_correct_phase)

# ----------------------------------------------------------------------------
# Plotting higher harmonic frequency functions
fig, ax_mag, ax_arg = al_plt.plot_mag_phase(
    freq_h=freq_Hs,
    magnitude = 20*np.log10(abs(Hs.transpose())),
    phase_deg = np.unwrap(np.angle(Hs_correct_phase.transpose()))/np.pi*180,
)
ax_mag.legend(('1st harmonic','2nd harmonic'), loc=3)
ax_mag.set_xlim(0, fs/2)
plt.show(block=False)
