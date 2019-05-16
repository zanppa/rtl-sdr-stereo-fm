# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:48:15 2019

This script creates a baseband signal for testing
stereo FM broadcast receiver.

The signal is what the FM demodulator would output and can be 
used to verify e.g. correct regeneration of clock signals.

Copyright (C) 2019 Lauri Peltonen
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

#####
# Configuration
write_wavs = False          # Write waveform files in the end
do_plots = True             # Plot signal waveforms etc
plot_preemp = False         # Plot frequency response of pre-emphasis filter (requires do_plots = True)

#####
# Helper functions

# From https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# Analog filter transfer function was found in
# Gnuradio documentation at
# https://github.com/gnuradio/gnuradio/blob/master/gr-analog/python/analog/fm_emph.py
def fm_preemp(data, fs, tau=75e-6):
    global plot_preemp

    zeros = [-1/tau]
    poles = [-1/(3.097e-6)]     # Roll-off at 40 kHz
    gain = 1                    # Gain is fixed later
    
    # First create digital counterpart from the analog filter
    z, p, k = signal.bilinear_zpk(zeros, poles, gain, fs)
    
    # Then convert it to sos type for more stability
    sos_filter = signal.zpk2sos(z, p, k)

    # Find gain at 0 Hz
    w, h = signal.sosfreqz(sos_filter, worN=[0], fs=fs)

    # Apply so much gain that 0 Hz = 0 dB
    # and re-create the filter
    sos_filter = signal.zpk2sos(z, p, k/np.abs(h[0]))

    if do_plots and plot_preemp:
        # Draw the filter frequency response
        w, h = signal.sosfreqz(sos_filter, worN=1500, fs=fs)
        plt.figure(10)
        plt.subplot(2, 1, 1)
        db = 20*np.log10(np.abs(h))
        plt.plot(w/np.pi, db)
        plt.ylim(-5, 35)
        plt.grid(True)
        plt.yticks([30, 20, 10, 0])
        plt.ylabel('Gain [dB]')
        plt.title('Frequency Response')
        plt.subplot(2, 1, 2)
        plt.plot(w/np.pi, np.angle(h))
        plt.grid(True)
        plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.ylabel('Phase [rad]')
        plt.xlabel('Normalized frequency (1.0 = Nyquist)')
        plt.show()

    return signal.sosfilt(sos_filter, data)


####
# Code starts here

# Configuration
fs = 384000     # Sampling rate
t_end = 1       # 1 second of data
samples = t_end * fs + 1

# Pilot signals
f_pilot = 19000.0         # FM pilot signal frequency
phase_pilot = 0.0       # in radians

# Left and right signal frequencies
f_left = f_pilot/30    # Left channel sawtooth
f_right = f_pilot/10  # Right channel sawtooth frequency

# RDS configuration
f_rds = 57000               # RDS carrier
phase_rds = phase_pilot
bps_rds = 1187.5          # RDS bps



plt.close("all")


# Generate base signals
time_array, Ts = np.linspace(0, t_end, samples, retstep = True)
left_array = .5 * signal.sawtooth(2. * np.pi * f_left * time_array)
right_array = .5 * signal.sawtooth(2. * np.pi * f_right * time_array)

# Low pass filter the channels @ 15 kHz
# Type I Chebyshev filter
lp_filter_b, lp_filter_a = signal.cheby1(11, .1, 15000. / (0.5 * fs), btype='lowpass', analog=False)
left_array = signal.lfilter(lp_filter_b, lp_filter_a, left_array)
right_array = signal.lfilter(lp_filter_b, lp_filter_a, right_array)


# Pre-emphasis before doing stereo calculations or anything
left_array = fm_preemp(left_array, fs, tau=75.0e-6)
right_array = fm_preemp(right_array, fs, tau=75.0e-6)

# Convert left and right to sum and difference for stereo encoding
mono_array = left_array + right_array       # Mono channel
stereo_array = left_array - right_array    # "Stereo" channel

pilot_array = np.sin(2. * np.pi * f_pilot * time_array + phase_pilot)
pilot2_array = np.sin(4. * np.pi * f_pilot * time_array + phase_pilot)




# Pseudo RDS-signal
rds_len = int(samples / (fs / bps_rds)) + 1       # Make sure this is longer than audio
rds_data = np.random.randint(0, high=2, size=rds_len)
rds_scale = fs / bps_rds

# Modulate
rds_array = np.empty_like(time_array)
for idx, t in enumerate(time_array):
    rds_array[idx] = np.sin(2. * np.pi * f_rds * t + phase_rds + np.pi*(1 - rds_data[int(idx // rds_scale)]))

rds_array = butter_bandpass_filter(rds_array, 55500, 58500, fs, order=5)




# Generate the signal to be FM modulated
signal_array = 0.9 * (.5 * mono_array + .5 * stereo_array * pilot2_array) + \
                0.1 * pilot_array +  \
                0.05 * rds_array
                
# Normalize all arrays so the plots look correct
norm_factor = 1. / np.max(np.abs(signal_array))
signal_array *= norm_factor
left_array *= norm_factor
right_array *= norm_factor
mono_array *= norm_factor
stereo_array *= norm_factor

# Do not normalize clocks as it's simpler if their amplitude is 1.0


# Plot everything
if do_plots:
    plt.figure(1)
    #plt.subplot(221)
    plt.title('Left and right')
    plt.xlabel('Time [s]')
    plt.plot(time_array[0:2048], left_array[0:2048], label='Left')
    plt.plot(time_array[0:2048], right_array[0:2048], label='Right')
    plt.legend(loc=1)
    
    plt.figure(2)
    #plt.subplot(222)
    plt.xlabel('Time [s]')
    plt.title('Sum (L+R, mono) and difference (L-R, "stereo")')
    plt.plot(time_array[0:2048], mono_array[0:2048], label='L+R')
    plt.plot(time_array[0:2048], stereo_array[0:2048], label='L-R')
    plt.legend(loc=1)
    
    plt.figure(3)
    plt.xlabel('Time [s]')
    plt.title('Signal to be modulated')
    plt.plot(time_array[0:2048], signal_array[0:2048], label='Signal')
    plt.axvspan(0.00155, 0.00190, alpha=0.3, color='red')
    
    plt.figure(4)
    #plt.subplot(223)
    plt.plot(time_array[0:2048], signal_array[0:2048], label='Signal')
    plt.plot(time_array[0:2048], pilot_array[0:2048], label='19 kHz clock')
    plt.plot(time_array[0:2048], pilot2_array[0:2048], label='38 kHz clock')
    plt.xlim([0.00155, 0.00190])
    plt.grid()
    plt.legend(loc=1)
    plt.title('Clock phase detail')
    
    # Calculate the spectrum and plot it
    plt.figure(5)
    #plt.subplot(224)
    signal_fft = np.fft.fft(signal_array)
    signal_fft_freqs = np.linspace(0.0, 1.0 / (2.0*Ts), samples//2) #np.fft.fftfreq(samples, d=Ts)
    plt.plot(signal_fft_freqs, 20. * np.log10(2.0/samples * np.abs(signal_fft[0:samples//2])), color='black')
    plt.xlim([0, 75000])
    plt.ylim([-120, 0])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.title('Spectrum')
    plt.axvspan(30, 15000, alpha=0.3, color='red')
    plt.axvspan(18500, 19500, alpha=0.3, color='green')
    plt.axvspan(23000, 53000, alpha=0.3, color='yellow')
    plt.axvspan(55000, 59000, alpha=0.3, color='orange')
    plt.text(7500, -10, 'L+R', horizontalalignment='center')
    plt.text(19000, -10, '19kHz pilot', horizontalalignment='center')
    plt.text(38000, -10, 'L-R', horizontalalignment='center')
    plt.text(57000, -10, 'RDS', horizontalalignment='center')
    
    #plt.figure(6)
    #plt.plot(time_array[0:2048], rds_array[0:2048])


if write_wavs:
    # Write the ("demodulated") signal to wav
    wavfile.write("fm_demod_test.wav", fs, signal_array)
    
    # Write the original tracks to wav
    wavfile.write("fm_orig_channels.wav", fs, np.stack([left_array, right_array], axis=1))
