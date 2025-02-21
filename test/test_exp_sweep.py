import audiolib.plotting as al_plt
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc_sp

from audiolib.signal_processing import ExpSweep

plt.close('all')

fs = 48000
f1 = 2e3
f2 = 20e3
dur = 5
apply_fade_to = 'both'

sweep = ExpSweep(
    fs=fs,
    f1=f1,
    f2=f2,
    approx_dur = dur,
)

t_sweep, s_sweep = sweep.get_sweep_signal()

fir = [.5, .5] # FIR Low-Pass with cutoff @ fs/2

fir = sc_sp.firwin(numtaps=100, cutoff=fs/4, fs=fs,)

y = sc_sp.lfilter(b = fir, a = [1, 0], x = s_sweep, )

t_hs, hs, freq_Hs, Hs = sweep.get_hhfrfs(
     y=y,
     n_harms=2,
)

fig, ax = al_plt.plot_rfft_freq(
    f=freq_Hs,
    data = 20*np.log10(abs(Hs.transpose())),
)
ax.legend(('1st harmonic','2nd harmonic'), loc=3)
ax.set_xlim(0, fs/2)
plt.show(block=False)

