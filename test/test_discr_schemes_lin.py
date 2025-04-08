import audiolib.signal_processing.state_space as st_sp
import audiolib.plotting as al_plt
import audiolib.elac as al_elac
import matplotlib.pyplot as plt
import numpy as np
import warnings

from audiolib.signal_processing import ExpSweep
from SynchSweptSine import SynchSweptSine

# ----------------------------------------------------------------------------
# General variables
fig_size = (6.4, 4.5)
freq = np.arange(1, 2e3, )
fs = 24000
Ts = 1/fs

sig_dur = 10
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
    f1 = 3
    f2 = 11e3
    fade_in = np.round(int(1*fs)) # s
    fade_out = np.round(int(0.01*fs)) # s
    sweep = ExpSweep(
        f1=f1,
        f2=f2,
        fs=fs,
        approx_dur=sig_dur,
        # apply_fade_to=None,
    )
    t, u_in = sweep.get_sweep_signal()
    sss = SynchSweptSine(f1=f1, f2=f2, T=sig_dur, fs=fs, fade=[fade_in, fade_out])
    u_in = sss.signal
    t = np.arange(0,np.round(fs*sig_dur-1)/fs,1/fs)  # time axis
    # t = t[:-1]
    # u_in = u_in[:-1]


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
heun_lin = st_sp.Heun(
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
heun_lin.run_over_input()


# ----------------------------------------------------------------------------
# Plotting Time

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
    data = 1e3*heun_lin.output_dict['x'][:plot_win_len],
    label = 'Heun Lin.',
    fig=fig_t,
    ax=ax_t,
)
al_plt.plot_time(
    t = t[:plot_win_len],
    data = u_in[:plot_win_len],
    label = 'Input signal',
    fig=fig_t,
    ax=ax_t,
)
ylims = max(abs(1e3*eb.output_dict['x']))
ax_t.set(
    ylim=[-ylims, ylims],
    ylabel='Displacement [mm]'
)


# ----------------------------------------------------------------------------
# Plotting Frequency
len_IR = 2**15
freq_irs = np.fft.rfftfreq(len_IR, 1/fs) # Freq. axis

h_xu_ef = sss.getIR(ef.output_dict['x'][:-1]) # the full impulse response
Hs_xu_ef = sss.separate_IR(h_xu_ef, N=2, n_samples=len_IR)    # separatef HHFRs

Hs_xu_ef_arg = Hs_xu_ef*np.exp(-1j*freq_irs*np.pi*len_IR/fs)

fig_x, ax_mag_x, ax_arg_x = al_plt.plot_mag_phase(
    freq_h = freq_irs,
    magnitude = 20*np.log10(abs(Hs_xu_ef.transpose())),
    phase_deg = np.angle(Hs_xu_ef_arg.transpose()), # *exp(-1j*freq*pi*nfft/hhir.samplerate)
    xscale='log',
    label='EF',
)

plt.show(block=False)