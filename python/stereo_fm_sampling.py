# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:22:47 2019

This script is used to show how it is possible
to demodulate stereo FM broadcast to left and 
right channels by sampling the raw signal 
at positive and negative peaks of the 38 kHz 
carrier.

Copyright (C) 2019 Lauri Peltonen
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Configuration
fs = 192000     # Sampling rate
t_end = 0.005       # seconds of data
samples = int(t_end * fs + 1)


plt.close("all")


# Generate base signals
time_array, Ts = np.linspace(0, t_end, samples, retstep = True)
left_array = np.random.normal(size=samples)
right_array = np.random.normal(size=samples)

# Low pass filter the channels @ 15 kHz
# Type I Chebyshev filter
lp_filter_b, lp_filter_a = signal.cheby1(11, .1, 3000. / (0.5 * fs), btype='lowpass', analog=False)
left_array = signal.lfilter(lp_filter_b, lp_filter_a, left_array)
right_array = signal.lfilter(lp_filter_b, lp_filter_a, right_array)

# Convert left and right to sum and difference for stereo encoding
mono_array = left_array + right_array       # Mono channel
stereo_array = left_array - right_array    # "Stereo" channel

# Carrier for difference channel modulation
# use arbitrary frequency that shows up nicely on the graph
carrier_array = np.sin(2. * np.pi * fs/16. * time_array)

# Generate the baseband signal
signal_array = (.5*mono_array + .5*stereo_array * carrier_array)

# Generate sampling markers
samples = int(t_end * fs/16. + 1)
sample_left_time = time_array[4::16]
sample_left_value = signal_array[4::16]
sample_right_time = time_array[12::16]
sample_right_value = signal_array[12::16]


# Draw ticks at carrier peaks
minor_ticks = np.arange(4./fs, t_end+4./fs, 8./fs)


# Plot the thing
plt.figure(1)
plt.plot(time_array, left_array, label='Left', color='blue')
plt.plot(time_array, right_array, label='Right', color='red')
plt.plot(time_array, carrier_array, label='Carrier', color='teal')
plt.plot(time_array, signal_array, label='Signal', color='black')
plt.plot(sample_left_time, sample_left_value, 'bo', label='Left sample points')
plt.plot(sample_right_time, sample_right_value, 'ro', label='Right sample points')
plt.grid()
plt.xticks(minor_ticks, [])
plt.legend(loc=1)
plt.title('Sampling')