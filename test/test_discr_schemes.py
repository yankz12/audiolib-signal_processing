import audiolib.signal_processing.state_space as st_sp
import audiolib.plotting as al_plt
import audiolib.elac as al_elac
import matplotlib.pyplot as plt
import numpy as np

from audiolib.signal_processing import ExpSweep

# ----------------------------------------------------------------------------
# General variables
fig_size = (6.4, 4.5)
freq = np.arange(1, 2e3, )
fs = 48000
Ts = 1/fs

sig_dur = .5
sig_type = 'sweep' # ['dirac', 'sine', 'sweep']

plot_win_dur = .4 # 200e-3 # 50ms [s]
plot_win_len = int(np.round(plot_win_dur*fs))

ref_vel = 1e-9
ref_displ = 1e-12


# ----------------------------------------------------------------------------
# Measurement signal definition
if sig_type == 'dirac':
    t = np.arange(0, sig_dur, 1/fs)
    u_in = np.zeros(len(t)) # Input
    u_in[10] = 1
elif sig_type == 'sine':
    t = np.arange(0, sig_dur, 1/fs)
    f = 20 # Hz
    plot_win_dur = 6/f
    plot_win_len = int(np.round(plot_win_dur*fs))
    u_in = np.sin(2*np.pi*f*t)
elif sig_type == 'sweep':
    f1 = 20
    f2 = 2000
    sweep = ExpSweep(
        f1=f1,
        f2=f2,
        fs=fs,
        approx_dur=sig_dur,
    )
    t, u_in = sweep.get_sweep_signal()
    t = t[:-1]
    u_in = u_in[:-1]


# ----------------------------------------------------------------------------
# System definition
Re = 4.07
Le = 0.5e-3 
Bl = 6.986  
Mms = 18.484e-3
Cms = 0.828e-3
Kms = 1/Cms
Rms = 0.565


# ----------------------------------------------------------------------------
# State Space Calculations
obs_order = [
    'i',
    'x',
    'v',
]
A = np.array(
    [
        [-Re/Le,    0,              -Bl/Le  ],
        [0,         0,              1       ],
        [Bl/Mms,    -Kms/Mms,       -Rms/Mms]
    ]
)
B = np.array([1/Le, 0, 0, ])

eb = st_sp.EulerBackward(
    A=A,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
)
ef = st_sp.EulerForward(
    A=A,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
)
bil = st_sp.Bilinear(
    A=A,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
)
ab = st_sp.AdamBashforth(
    A=A,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
    order=3,
)
heun = st_sp.Heun(
    A=A,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
)

eb.run_over_input()
ef.run_over_input()
bil.run_over_input()
ab.run_over_input()
heun.run_over_input()

fig_t, ax_t = al_plt.plot_time(
    t = t[:plot_win_len],
    data = 1e3*ab.output_dict['x'][:plot_win_len],
    label = 'AB 3rd',
)
al_plt.plot_time(
    t = t[:plot_win_len],
    data = 1e3*ef.output_dict['x'][:plot_win_len],
    label = 'FW Euler',
    fig=fig_t,
    ax=ax_t,
)
al_plt.plot_time(
    t = t[:plot_win_len],
    data = 1e3*eb.output_dict['x'][:plot_win_len],
    label = 'BW Euler',
    fig=fig_t,
    ax=ax_t,
)
al_plt.plot_time(
    t = t[:plot_win_len],
    data = 1e3*bil.output_dict['x'][:plot_win_len],
    label = 'Bil.',
    fig=fig_t,
    ax=ax_t,
)
al_plt.plot_time(
    t = t[:plot_win_len],
    data = 1e3*heun.output_dict['x'][:plot_win_len],
    label = 'Heun',
    fig=fig_t,
    ax=ax_t,
)
ylims = max(abs(1e3*eb.output_dict['x']))
ax_t.set(
    ylim=[-ylims, ylims],
    ylabel='Displacement [mm]'
)
plt.show(block=False)