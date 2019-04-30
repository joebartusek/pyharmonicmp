import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import interpolation
from importlib import reload
from scipy.io.wavfile import read, write
import sounddevice as sd
import matplotlib.mlab as mlab
from multiprocessing import Pool
from functools import partial
__spec__ = None

def get_subspace(window, unw_ss):
    ss = []
    window_inds = window > 0
    trunc_window = window[window_inds]
    # trunc_window_shift = np.argmax(window > 0)
    # shift = np.argmax(window) - (len(window) // 2)
    for sig in unw_ss:
        trunc_sig = sig[window_inds]
        trunc_atom = (trunc_sig * trunc_window)
        trunc_atom = trunc_atom / np.linalg.norm(trunc_atom)
        atom = np.zeros(len(window))
        atom[window_inds] = trunc_atom
        ss.append(atom)
    return np.array(ss)

def Q_corr(subspace, residual):
    result = 0
    for x in subspace:
        result += np.abs(np.dot(x, residual)) ** 2
    return result

def artificial_sound(windows, uw_subspaces, num, noise_amt = 0.05):
    y = np.zeros(npts)
    windows_use = [np.random.choice(len(windows), num), np.random.choice(len(uw_subspaces), num)]
    for wind_ind, uw_ind in np.array(windows_use).T:
        ss = get_subspace(windows[wind_ind], uw_subspaces[uw_ind]).T
        coef = np.random.uniform(0, 1, len(ss[0]))
        y = y + np.dot(ss, coef)
    y += np.random.uniform(-1 * noise_amt, noise_amt, y.shape)
    return y

def corr_from_inds(inds, windows, uw_subspaces, resid):
    window = windows[inds[1]]
    signals = uw_subspaces[inds[0]]
    subspace = get_subspace(window, signals)
    score = Q_corr(subspace, resid)
    return (inds, score)

if __name__ == '__main__':
    in_fname = "cello-a3.wav"
    a = read(in_fname)
    sound = np.array(a[1],dtype=float)
    if len(sound.shape) > 1:
        sound = sound[:, 0]

    sound = sound / max(sound)
    base_hz = 32.7
    octaves = 1

    sr = 44100
    num_harmonics = 20
    corr_thresh_quantile = 0.90
    npts = int(sr * 2)
    sound = sound[:npts]
    domain = range(npts)
    gauss_eps = 0.001
    resid_eps = 0.05
    num_workers = 3
    max_iterations = 50

    u_grid = np.linspace(0, npts, 50)
    u_spacing = u_grid[1] - u_grid[0]
    s_grid = np.geomspace(u_spacing / 2, u_spacing * 2, 3)
    s_windows = []
    for s in s_grid:
        window = signal.gaussian(npts, s)
        window[window < gauss_eps] = 0
        s_windows.append(window)

    f0_grid = (2 * np.pi / sr) * np.geomspace(32.7, 32.7 * (2 ** octaves), 1 + (octaves * 12))

    # define harmonic subspaces
    # project onto each harmonic subspace
    # perform MP strictly within each subspace

    print('shifting/scaling windows...')
    shifted_windows = []
    for u, s_window in iter.product(u_grid, s_windows):
        center_dist = (len(s_window) // 2) - int(u)
        new_window = interpolation.shift(s_window, center_dist)
        shifted_windows.append(new_window)

    unwindowed_subspaces = []
    print('creating base harmonic signals...')
    for i, f in enumerate(f0_grid):

        centered_domain = np.array(domain) - (len(domain) // 2)
        main_signal = [np.cos(f * centered_domain * n) for n in range(1, num_harmonics + 1)]
        main_signal += [np.sin(f * centered_domain * n) for n in range(1, num_harmonics + 1)]
        unwindowed_subspaces.append(main_signal)

    # y = artificial_sound(shifted_windows, unwindowed_subspaces, 10, noise_amt = 0.00)
    y = sound

    resid = y
    reconstruction = np.zeros(npts)
    found_atoms = []
    scores_array = []
    subspace_inds = iter.product(range(len(unwindowed_subspaces)), range(len(shifted_windows)))
    subspace_inds = np.array(list(subspace_inds))
    inds_rejected = np.zeros(len(subspace_inds))
    threshold_score = 0
    subspace_evaluations = []
    resid_norms = []
    pool = Pool(processes=num_workers)

    print('beginning pursuit...')
    for it in range(max_iterations):

        scores = np.zeros(len(subspace_inds))
        best_inds = False
        active_subspace_inds = subspace_inds[inds_rejected != 1]

        # compute correlation score against current partial: now in parallel!
        corr_partial = partial(corr_from_inds,
            windows=shifted_windows,
            uw_subspaces=unwindowed_subspaces,
            resid=resid)
        active_scores = pool.map(corr_partial, active_subspace_inds)

        # get the index of the best subspace and the scores
        best_inds = max(active_scores, key=lambda x: x[1])[0]
        scores[inds_rejected == 0] = [x[1] for x in active_scores]

        # for all scores that are super low, set their subspaces to be ignored
        subspace_evaluations.append(sum(inds_rejected == 0))
        if threshold_score == 0:
            threshold_score = np.quantile(scores[scores > 0], corr_thresh_quantile)
        inds_rejected[scores < threshold_score] = 1

        if all(inds_rejected):
            print('re-initializing dictionary...')
            threshold_score = 0
            inds_rejected[:] = 0

        # once the most-correlated subspace has been found, compute projection onto that subspace
        select_subspace = get_subspace(shifted_windows[best_inds[1]], unwindowed_subspaces[best_inds[0]])
        new_atom = np.zeros(npts)

        for mbr in select_subspace:
            factor = np.dot(mbr, resid)
            new_atom = new_atom + (factor * mbr)

        found_atoms.append(new_atom)

        reconstruction = reconstruction + new_atom
        new_resid = resid - new_atom
        resid_norm = np.linalg.norm(new_resid) / np.linalg.norm(y)
        resid_norms.append(resid_norm)

        print('iter {}, resid norm {}, excluded {}'.format(it, resid_norm, sum(inds_rejected)))

        resid = new_resid

        if resid_norm < resid_eps:
            print('ending - residual below target norm')
            break

    plt.clf()
    plt.subplot(221)
    plt.plot(domain, y)
    plt.subplot(222)
    plt.plot(domain, reconstruction)
    plt.plot(domain, resid, c='red')
    plt.subplot(223)
    plt.specgram(y, NFFT=256, Fs=sr, scale='dB')
    plt.subplot(224)
    plt.specgram(reconstruction, NFFT=256, Fs=sr, scale='dB')
    plt.show()


    # sd.play(np.concatenate([y, np.zeros(sr), reconstruction, np.zeros(sr), resid]), sr)
    # write('harpsichord-1.wav', sr, np.concatenate([y, np.zeros(sr), reconstruction]))
    # write('harpsichord-allatoms.wav', sr, np.concatenate(found_atoms))
    # #
    # plt.clf()
    # plt.plot(resid_norms_0noise)
    # plt.plot(resid_norms_01noise)
    # plt.plot(resid_norms_03noise)
    # plt.plot(resid_norms_05noise)
    # plt.xlabel('Iterations')
    # plt.ylabel('Norm Residual / Norm Input')
    # plt.legend(['No noise added', 'Added white noise * 0.01', 'Added white noise * 0.03', 'Added white noise * 0.05'])
    # plt.savefig('noisenorms.png', bbox_inches='tight', dpi=300)
    # plt.show()
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
