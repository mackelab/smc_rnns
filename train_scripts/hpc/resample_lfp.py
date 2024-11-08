import numpy as np
from neo.io.neuroscopeio import NeuroScopeIO
from scipy.signal import resample

# we are using data from: https://crcns.org/data-sets/hc/hc-11/about-hc-11
# Grosmark, A.D., and Buzsáki, G. (2016). Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences. Science 351, 1440–1443.
# Chen, Z., Grosmark, A.D., Penagos, H., and Wilson, M.A. (2016). Uncovering representations of sleep-associated hippocampal ensemble spike activity. Sci. Rep. 6, 32193.

# We are also using code from: https://github.com/zhd96/pi-vae/blob/main/examples/pi-vae_rat_data.ipynb
# Zhou, D., Wei, X.
# Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE.
# NeurIPS 2020. https://arxiv.org/abs/2011.04798

# path to the preprocessed matfile from the pi-VAE authors:
# https://drive.google.com/drive/folders/1lUVX1IvKZmw-uL2UWLxgx4NJ62YbCwMo?usp=sharing

path_mat = "../../data_untracked/Achilles_10252013_sessInfo.mat"

# path to the eeg data from: https://crcns.org/data-sets/hc/hc-11/about-hc-11
path_lfp = "../../../Achilles_10252013/Achilles_10252013.eeg"
LFP_fs = 100
resample_rate = 1250 / LFP_fs
reader = NeuroScopeIO(filename=path_lfp)
seg = reader.read_segment(lazy=False)
t, c = np.shape(seg.analogsignals[0])
ds = []
for i in range(c):
    print(i)
    lfp = np.array(seg.analogsignals[0][:, i])
    # resample
    n_samples = int(len(lfp) / resample_rate)
    lfp_ds = resample(lfp, n_samples)
    ds.append(lfp_ds)
LFP = np.array(ds)
np.save("../../data_untracked/Achilles_10252013_lfp_100Hz.npy", LFP)
print("LFP shape " + str(LFP.shape))
