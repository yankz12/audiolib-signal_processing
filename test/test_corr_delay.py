import audiolib.signal_processing as al_sp
import audiolib.plotting as al_plt

import numpy as np
import matplotlib.pyplot as plt

# Create Sine signal, window sine, apply zero-padding
def create_windowed_sinus_with_zero(t, start_time, end_time, freq, ramp_time):
    signal = np.zeros_like(t)
    active_idx = np.where((t >= start_time) & (t < end_time))[0]
    pure_sine = np.sin(2 * np.pi * freq * t[:len(active_idx)])
    # Indizes für den Zeitraum, in dem das Signal aktiv ist
    # Sinus innerhalb des aktiven Bereichs
    signal[active_idx] = al_sp.apply_window(
        pure_sine,
        fs=int(1/(t[1] - t[0])),
        win_type='hann',
        win_dur=2*ramp_time,
    )
    return signal

# Parameters
fs = 48000
duration = 1.0
n_samples = int(fs * duration)

# Sinus-Parameter
f = 100
period = 1 / f
n_periods = 40
sine_duration = n_periods * period 
win_ramp_time = 0.05
delay = 0.0527

# Zeitvektor
t = np.linspace(0, duration, n_samples, endpoint=False)

# Start- und Endzeit des Sinussignals
start_time = 0
end_time = sine_duration

# Erzeuge die beiden Signale
signal1_windowed = create_windowed_sinus_with_zero(
    t, start_time, end_time, f, win_ramp_time
)

# Verzögerung von 0.05 Sekunden für das zweite Signal

start_time_2 = delay
end_time_2 = delay + sine_duration
signal2_windowed = create_windowed_sinus_with_zero(
    t, start_time_2, end_time_2, f, win_ramp_time
)

lag = al_sp.get_delay_via_crosscorr(
    x = signal1_windowed,
    y = signal2_windowed,
    fs=fs,
    plot = True,
)

fig, ax = al_plt.plot_time(
    t,
    signal1_windowed,
    label='Signal 1',
)
al_plt.plot_time(
    t,
    signal2_windowed,
    label='Signal 2',
    fig=fig,
    ax=ax,
)
al_plt.plot_time(
    t,
    np.roll(signal2_windowed, int(lag*fs)),
    label='Signal 2 shifted',
    fig=fig,
    ax=ax,
    ls='--'
)
sign = '+' if lag > 0 else '-'
title_str = f'Signal 1 = Signal 2 {sign} {abs(lag)}s'
ax.set_title(title_str)
plt.show(block=False)
