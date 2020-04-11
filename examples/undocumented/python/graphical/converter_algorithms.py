"""
Blind Source Separation using the FastICA, FFSep, Jade, JediSep, SOBI, UWedgeSep algorithms with Shogun

Based on the example from scikit-learn
http://scikit-learn.org/

Kevin Hughes 2013
"""

import matplotlib.pyplot as plt
import numpy as np
import shogun as sg

# Generate a sample data
np.random.seed(0)
NUM_SAMPLES = 2000


def get_signals_matrix(time_start, time_stop, signal_fun1, signal_fun2):
    time = np.linspace(time_start, time_stop, NUM_SAMPLES)

    S = np.c_[signal_fun1(time), signal_fun2(time)]
    S += 0.2 * np.random.normal(size=S.shape)  # add a noise

    # Standardize the data
    S /= S.std(axis=0)
    return S.T


for converter in ['FastICA', 'FFSep', 'Jade', 'JediSep', 'SOBI', 'UWedgeSep']:
    if converter == 'UWedgeSep':
        S_ = get_signals_matrix(0, 1,
                                # Source Signals
                                lambda time: np.sin(2 * 3.14 * 55 * time),      # sin wave
                                lambda time: np.cos(2 * 3.14 * 100 * time),     # cos wave
                                )
        # A mixing matrix
        A = np.array([[1, 0.85], [0.55, 1]])
    else:
        S_ = get_signals_matrix(0, 10,
                                lambda time: np.sin(2 * time),          # sin wave
                                lambda time: np.sign(np.sin(3 * time))  # square wave
                                )
        A = np.array([[1, 0.5], [0.5, 1]])

    # Mix the signals
    X = np.dot(A, S_)
    mixed_signals = sg.features(X)

    # Separating
    transformer = sg.transformer(converter)
    transformer.fit(mixed_signals)
    signals = transformer.transform(mixed_signals)
    S_ = signals.get('feature_matrix')
    A_ = transformer.get('mixing_matrix')

    # Plot the results
    plt.figure(converter)
    plt.subplot(3, 1, 1)
    plt.plot(S_.T)
    plt.title('True Sources')
    plt.subplot(3, 1, 2)
    plt.plot(X.T)
    plt.title('Mixed Sources')
    plt.subplot(3, 1, 3)
    plt.plot(S_.T)
    plt.title('Estimated Sources')
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
plt.show()
