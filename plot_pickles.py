import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
import numpy as np

# in_fname = "horn-a#3-50thresh.wav"
# a = read(in_fname)
# sound = np.array(a[1],dtype=float)
#
# 
# Pxx, freq, t = mlab.specgram(sound, Fs=a[0])
# plt.clf()
# ha = plt.subplot(111)
# ha.pcolor(t, freq, Pxx)
# ha.set_yscale('log')
# plt.savefig('asdf.png')


bassoon = pickle.load(open('bassoon-a#3.wav-resids.pik', 'rb'))
clarinet = pickle.load(open('clarinet-b3.wav-resids.pik', 'rb'))
cello = pickle.load(open('cello-a3.wav-resids.pik', 'rb'))
horn = pickle.load(open('horn-a#3.wav-resids.pik', 'rb'))
oboe = pickle.load(open('oboe-a#3.wav-resids.pik', 'rb'))

plt.clf()
plt.plot(bassoon)
plt.plot(clarinet)
plt.plot(cello)
plt.plot(oboe)
plt.plot(horn)
plt.xlabel('Iterations')
plt.ylabel('Norm Residual / Norm Input')
plt.legend(['bassoon', 'clarinet', 'cello', 'horn', 'oboe'])
plt.savefig('instrumentnorms.png', bbox_inches='tight', dpi=300)
plt.show()



    #
    # plt.clf()
    # plt.plot(resid_norms_50)
    # plt.plot(resid_norms_75)
    # plt.plot(resid_norms_90)
    # plt.plot(resid_norms_95)
    # plt.xlabel('Iterations')
    # plt.ylabel('Norm Residual / Norm Input')
    # plt.legend(['Threshold at 50%', 'Threshold at 75%', 'Threshold at 90%', 'Threshold at 95%'])
    # plt.savefig('residualnorms.png', bbox_inches='tight', dpi=300)
    # plt.show()
    #
    # plt.clf()
    # plt.plot(np.cumsum(subspace_evaluations_50))
    # plt.plot(np.cumsum(subspace_evaluations_75))
    # plt.plot(np.cumsum(subspace_evaluations_90))
    # plt.plot(np.cumsum(subspace_evaluations_95))
    # plt.xlabel('Iterations')
    # plt.ylabel('Num. Evaluations of correlation function')
    # plt.legend(['Threshold at 50%', 'Threshold at 75%', 'Threshold at 90%', 'Threshold at 95%'])
    # plt.savefig('evalfunc.png', bbox_inches='tight', dpi=300)
    # plt.show()
