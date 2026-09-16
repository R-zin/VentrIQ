import numpy as np
from scipy.signal import butter,filtfilt

class signal_preprocessing:
    def __init__(self):
        pass

    def bandpass_filter(self,ecg,lowcut=0.5,highcut=40.0,fs=250):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b,a = butter(4,[low,high],'low')
        return filtfilt(b,a,ecg,axis=-1)

    def normalize(self,ecg):
        mean = np.mean(ecg,axis=-1,keepdims=True)
        std = np.std(ecg,axis=-1,keepdims=True)
        return ecg-mean/(std+1e-8)
    def preprocess_ecg(self,ecg):
        ecg = np.asarray(ecg,dtype=np.float32)
        if not np.isfinite(ecg).all():
            return None
        ecg = self.bandpass_filter(ecg,lowcut=0.5,highcut=40.0,fs=250)
        ecg = self.normalize(ecg)
        return ecg