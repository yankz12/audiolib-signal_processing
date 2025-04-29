import audiolib.signal_processing.state_space as st_sp
import audiolib.plotting as al_plt
import audiolib.elac as al_elac
import matplotlib.pyplot as plt
import numpy as np
import warnings

from audiolib.signal_processing import ExpSweep

def non_lin_0(non_lin_x, non_lin_y, cur_x, ):
    # TODO: Move to state space class
    cur_idx = np.argmin(np.abs(cur_x - non_lin_x))
    cur_val = non_lin_y[cur_idx]
    cur_val_out_range = (
        (cur_x < 0 and cur_x < 2*non_lin_x[0]) or
        (cur_x > 0 and cur_x > 2*non_lin_x[-1])
    )
    if cur_val_out_range:
        warnings.warn(
            'Observation Value twice as big as range of non-linear ' +
            'value table: Extend table or reduce input signal amplitude!'
        )
    return cur_val

def non_lin_kms(tmp_output, ):
    return -Kms*(1 + 1e5*tmp_output['x']**2)/Mms

# ----------------------------------------------------------------------------
# General variables
fig_size = (6.4, 4.5)
fs = 48000
Ts = 1/fs

sig_dur = 3
sig_type = 'sweep' # ['dirac', 'sine', 'sweep']
n_harms = 3
apply_fade_to = None
len_irs = 2**12


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
    f1 = 5
    f2 = 2000
    sweep = ExpSweep(
        f1=f1,
        f2=f2,
        fs=fs,
        approx_dur=sig_dur,
        apply_fade_to=apply_fade_to,
        num_harmonics=n_harms,
        len_irs=len_irs,
    )
    t, u_in = sweep.get_sweep_signal()
    t = t[:-1]
    u_in = 10*u_in[:-1]


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
A_lin = np.array(
    [
        [-Re/Le,    0,              -Bl/Le  ],
        [0,         0,              1       ],
        [Bl/Mms,    -Kms/Mms,       -Rms/Mms]
    ]
)
A_nonlin = np.array(
    [
        [-Re/Le,    0,              -Bl/Le  ],
        [0,         0,              1       ],
        [Bl/Mms,    non_lin_kms,    -Rms/Mms]
    ]
)

B = np.array([1/Le, 0, 0, ])

heun_nonlin = st_sp.Heun(
    A=A_nonlin,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
)
heun_lin = st_sp.Heun( 
    A=A_lin,
    B=B,
    input_sig=u_in,
    input_time=t,
    obs_order = obs_order,
)
heun_nonlin.run_over_input()
heun_lin.run_over_input()

# ----------------------------------------------------------------------------
# Sweep Post-Processign
t_hs_non_lin, hs_non_lin, freq_Hs_non_lin, Hs_non_lin, _ = sweep.get_hhfrfs(
    y=heun_nonlin.output_dict['x'],
)
Hs_non_lin = sweep.revert_delay(Hs_non_lin)

t_hs_lin, hs_lin, freq_Hs_lin, Hs_lin, _ = sweep.get_hhfrfs(
    y=heun_lin.output_dict['x'],
)
fix_hs, ax_hs = al_plt.plot_time(
    t = t_hs_non_lin,
    data = np.fft.fftshift(hs_non_lin).transpose(),
)
# Hs_lin = sweep.revert_delay(Hs_lin)

fig_t, ax_t = al_plt.plot_time(
    t = t, #[:plot_win_len],
    data = 1e3*heun_nonlin.output_dict['x'][:-1], #[:plot_win_len],
    label = 'Heun Non-Lin.',
)
al_plt.plot_time(
    t = t, #[:plot_win_len],
    data = 1e3*heun_lin.output_dict['x'][:-1],# [:plot_win_len],
    label = 'Heun Lin.',
    fig=fig_t,
    ax=ax_t,
)
ylims = max(abs(1e3*heun_lin.output_dict['x']))
ax_t.set(
    ylim=[-ylims, ylims],
    ylabel='Displacement [mm]'
)


fig_f, ax_mag_f, ax_arg_f = al_plt.plot_mag_phase(
    freq_h = freq_Hs_lin,
    magnitude = 20*np.log10(np.abs(Hs_non_lin)).transpose(),
    phase_deg = np.angle(Hs_non_lin.transpose())/np.pi*180,
    xscale='log',
)
ax_arg_f.set(
    ylim=[-180, 180],
    xlim=[f1, f2]
)
# al_plt.plot_mag_phase(
#     freq_h = freq_Hs_lin,
#     magnitude = 20*np.log10(np.abs(Hs_lin).transpose()),
#     phase_deg = np.unwrap(np.angle(Hs_lin.transpose()))/np.pi*180,
#     fig=fig_f,
#     ax_mag=ax_mag_f,
#     ax_arg=ax_arg_f,
# )
ax_mag_f.legend(
    ('Non-Lin.', 'Lin.', ),
)

plt.show(block=False)