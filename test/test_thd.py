import audiolib.signal_processing as al_sc
import matplotlib.pyplot as plt
import numpy as np


dur = 2
fs = 48000
t = np.linspace(0, dur, int(dur*fs))
f0 = 103.2
num_harms = 4

x = np.sin(2*np.pi*f0*t)
x += 0.1*np.sin(2*np.pi*2*f0*t)
x += 0.05*np.sin(2*np.pi*3*f0*t)
x += 0.05*np.sin(2*np.pi*4*f0*t)
x += 0.05*np.sin(2*np.pi*5*f0*t)

thd = al_sc.thd_from_time_sig(
    x,
    fs,
    num_harms = num_harms,
    tol_hz = 10,
    plot_spec = True,
)
thd_by_hand = 100*np.sqrt(0.1**2 + 0.05**2 + 0.05**2 + 0.05**2)

print(f'THD by Repo: {np.round(thd, 2)} %')
print(f'THD by Hand: {np.round(thd_by_hand, 2)} %')

plt.show(block=False)
