import numpy as np 
import scipy.signal as scsp

from scipy.fftpack import fftshift

def get_rfft_power_spec(x, fs, Nfft=None):
    if Nfft is None:
        Nfft = len(x)
    freq = np.arange(Nfft/2+1)/(Nfft/2+1)*fs/2
    Sxx = np.abs(np.fft.rfft(x, Nfft) / Nfft)**2
    return freq, Sxx

def get_ir_from_rfft(freq, cplx_data_spec, fs, nfft):
    centered_ir = fs*fftshift(np.fft.irfft(cplx_data_spec))
    t = np.arange(-int(nfft/2),int(nfft/2))/nfft
    return t, centered_ir

def get_ir_from_rawdata(t, x, y, fs, nfft):
    # TODO: Implement
    print('To be implemented.')
    return

def get_msc(sig_0, sig_1, fs, blocklen, ):
    # TODO: Test
    print('Not tested, use with caution!')
    freq_msc, msc = scsp.coherence(
        sig_0,
        sig_1,
        fs,
        nperseg=blocklen,
    )
    return freq_msc, msc
