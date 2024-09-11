import numpy as np
import random

G = np.array([[1, 0, 0, 0, 1, 0, 1],
              [0, 1, 0, 0, 1, 1, 1],
              [0, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 1, 0, 1, 1]])

# Parity-check matrix for Hamming(7,4) code
H = np.array([[1, 1, 1, 0, 1, 0, 0],
              [0, 1, 1, 1, 0, 1, 0],
              [1, 1, 0, 1, 0, 0, 1]])

def hamming_encode(bits):
    """Encode 4 bits using the Hamming (7,4) code."""
    # Encode the 4-bit data using the generator matrix G
    return np.dot(bits, G) % 2

def bpsk_modulate(bits):
    """Modulate the bits using BPSK: 0 -> -1, 1 -> +1."""
    return -2*(bits - 0.5)

def awgn_channel(signal, sigma):
    """Add AWGN noise to the signal based on the specified SNR."""
    noise = np.sqrt(sigma) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return signal + noise.real



def hamming_par(SNR):
    rc = 4/7
    eb_N0 = 2*(rc)*(10**(SNR / 10))
    sigma = 1/eb_N0
    total_errors = 0
    total_bits = 0
    lookup_table = {}
    for i in range(16):  # There are 2^4 = 16 possible 4-bit combinations
        data_bits = np.array([int(x) for x in np.binary_repr(i, width=4)])  # Convert integer to binary array
        encoded_bits = np.dot(data_bits, G) % 2  # Encode using Hamming (7,4)
        tx_signal = bpsk_modulate(encoded_bits)  # Modulate using BPSK
        lookup_table[tuple(data_bits)] = (encoded_bits, tx_signal)

    while total_errors < 1250:  # Collect at least 100 errors
        # Generate random 4-bit data words
        data_bits = random.choice(list(lookup_table.keys()))
        encoded_bits, tx_signal = lookup_table[data_bits]
        rx_signal = awgn_channel(tx_signal, sigma)

        # Decode using belief propagation
        decoded_bits = belief_propagation_decode(rx_signal,sigma, H)
        # Calculate bit errors between original data bits and decoded bits
        decoded_data = decoded_bits.reshape(-1, 7)[:, :4]  # Extract the 4 original data bits
        
        bit_errors = np.sum(np.abs(data_bits - decoded_data))
        # Update totals
        total_errors = total_errors + bit_errors 
        total_bits += 4
        if (total_bits % 40000000 == 0):
            print("Errors: ",total_errors)
            print("Bits", total_bits)

    return total_errors, total_bits

def bpsk_demodulate(received_signal):
    """Demodulate the BPSK signal: values > 0 map to 1, otherwise to 0."""
    return received_signal < 0

def belief_propagation_decode(rx, sigma, H, max_iterations=10):
    """Decode a single 7-bit block using belief propagation."""
    n = H.shape[1]  # Number of bits per block (7 for Hamming(7,4))
    # Initialize LLRs
    log_sigma = (-4 * rx) / (2 * sigma)
    llr = log_sigma.reshape(1, n)  # Reshape to a single block
    # Initialize message matrices
    m_check_to_code = np.zeros(H.shape)
    m_code_to_check = np.copy(llr)
    for iteration in range(max_iterations):
        block_llr = llr[0]  # Single block
        # Check node update
        for i in range(H.shape[0]):  # Iterate over check nodes
            parity_row = H[i]
            check_indices = np.where(parity_row == 1)[0]
            tanh_values = np.tanh(m_code_to_check[0, check_indices] / 2)  # Precompute tanh for all relevant indices
            for j in check_indices:
                # Create a mask to exclude the current index j
                mask = check_indices != j
                # Get the product of the remaining tanh values
                product = np.prod(tanh_values[mask])
                # Clip the product to avoid numerical instability
                product = np.clip(product, -0.999999999999, 0.9999999999)
                # Update m_check_to_code with the result
                m_check_to_code[i, j] = 2 * np.arctanh(product)

            if (iteration + 1 != max_iterations):
            # Code node message update
                for j in range(n):
                    check_nodes = np.where(H[:, j] == 1)[0]
                    m_code_to_check[0, j] = block_llr[j] + np.sum(m_check_to_code[check_nodes, j]) 
                    decoded_bits = m_code_to_check.flatten() > 0
                    int_bits = decoded_bits.astype(int)
                    zer = np.dot(H, int_bits) % 2
                    if (np.all(zer == 0)):
                        return int_bits
                    m_code_to_check[0, j] = m_code_to_check[0, j] - m_check_to_code[i, j]
            else:
                for j in range(n):
                    connected_check_nodes = np.where(H[:, j] == 1)[0]
                    m_code_to_check[0, j] = np.sum([m_check_to_code[i, j] for i in connected_check_nodes]) + llr[0, j]
                    decoded_bits = m_code_to_check.flatten() > 0
                    int_bits = decoded_bits.astype(int)
                    return int_bits

    # Final LLR update
    decoded_bits = m_code_to_check.flatten() > 0
    return decoded_bits.astype(int)