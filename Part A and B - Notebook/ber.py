import numpy as np
from scipy.special import erfc
from scipy.special import expit as sigmoid  # Sigmoid function for stability in LLR updates
from scipy.sparse import csr_matrix
import random

def ber_par(SNR):
    tote = 0 
    totb = 0
    N0 = 1 / 10**(SNR/10)
    lookup_table = {}
    for i in range(16):  # There are 2^4 = 16 possible 4-bit combinations
        rbits = np.array([int(x) for x in np.binary_repr(i, width=4)]) 
        tx_signal = -2*(rbits - 0.5)
        lookup_table[tuple(rbits)] = (tx_signal)
    while tote < 1250:
        rbits = random.choice(list(lookup_table.keys()))
        tx_signal = lookup_table[rbits]

        # Add AWGN noise
        # Noise level (based on SNR)
        noise = np.sqrt(N0/2)*(np.random.randn(4) + 1j*np.random.randn(4))
        rx = tx_signal + noise.real  # Received signal (with noise)
        # BPSK demodulation: > 0 -> 1, < 0 -> 0
        rx2 = rx < 0
        
        # Calculate bit errors
        diff = np.abs(rbits - rx2)
        tote += np.sum(diff)  # Total errors
        totb += 4 # Total bits generated
    return tote, totb
