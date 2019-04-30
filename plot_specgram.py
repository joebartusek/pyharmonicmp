import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import numpy as np

in_fname = "horn-a#3-50thresh.wav"
a = read(in_fname)
sound = np.array(a[1],dtype=float)


Pxx, freq, t = mlab.specgram(sound, Fs=a[0]) # Other options, such as NFFT, may be passed
                                # to `specgram` here.

plt.clf()
ha = plt.subplot(111)
ha.pcolor(t, freq, Pxx)
ha.set_yscale('log')
plt.savefig('asdf.png')
