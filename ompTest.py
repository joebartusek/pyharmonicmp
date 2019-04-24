
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import interpolation
from importlib import reload
from scipy.io.wavfile import read

a = read("shenai.wav")
sound = np.array(a[1],dtype=float)
sound = sound / max(sound)

sr = 44100
gauss_eps = 0.001

num_harmonics = 10
npts = sr * 1
sound = sound[:npts]
domain = range(npts)

u_grid = np.linspace(0, npts, 50)
u_spacing = u_grid[1] - u_grid[0]

s_grid = np.geomspace(sr / 100, sr / 25, 2)
s_windows = []
for s in s_grid:
    window = signal.gaussian(npts, s)
    window[window < gauss_eps] = 0
    s_windows.append(window)

f0_grid = np.geomspace(65.406 * 4, 65.406 * 8, 12) * (2 * np.pi / sr)

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
print('creating harmonic subspaces...')
for i, f in enumerate(f0_grid):

    centered_domain = np.array(domain) - (len(domain) // 2)
    main_signal = [np.cos(f * domain * n) for n in range(1, num_harmonics + 1)]
    unwindowed_subspaces.append(main_signal)
    # for w in shifted_windows:
    #     subspace = [w * sig for sig in main_signal]
    #     subspace = [x / np.linalg.norm(x) for x in subspace]
    #     h_subspaces.append(subspace)

def get_subspace(window, unw_ss):
    ss = []
    for ss in unw_ss:
        atom = (unw_ss * window)
        atom = atom / np.linalg.norm(atom)
        ss.append(atom)
    return ss

def Q_corr(subspace, residual):
    result = 0
    for x in subspace:
        result += np.abs(np.dot(x, residual)) ** 2
    return result

# create artificial data
def artificial_sound(h_subspaces, num):
    y = np.zeros(npts)
    subspaces_use = np.random.choice(len(h_subspaces), num)
    for s_ind in subspaces_use:
        ss = np.array(h_subspaces[s_ind]).T
        coef = np.zeros(len(ss[0]))
        coef[np.random.randint(len(coef))] = np.random.uniform(0, 1)
        y = y + np.dot(ss, coef)
    return y

# y = artificial_sound(h_subspaces, 15)
y = sound

resid = y
reconstruction = np.zeros(npts)
found_atoms = []
subspace_inds = list(iter.product(range(len(shifted_windows)), range(len(unwindowed_subspaces))))

for it in range(5):

    scores = np.zeros(len(subspace_inds))
    best_subspace = False

    # compute correlation function Q between residual and every subspace in h_subspaces
    for i, inds in enumerate(subspace_inds):
        window = shifted_windows[inds[0]]
        signals = unwindowed_subspaces[inds[1]]
        subspace = get_subspace(window, signals)
        score = Q_corr(subspace, resid)
        if score > max(scores):
            best_subspace = subspace
        scores[i] = score

    # once the most-correlated subspace has been found, compute projection onto that subspace
    select_subspace = best_subspace
    new_atom = np.zeros(npts)

    for mbr in select_subspace:
        new_atom = new_atom + (np.dot(mbr, resid) * mbr)

    found_atoms.append(new_atom)
    reconstruction = reconstruction + new_atom
    new_resid = resid - new_atom

    print('iter {}, resid norm {}'.format(it, np.linalg.norm(new_resid)))

    resid = new_resid

#result = omp.omp(X, y, nonneg=False, maxit=50, orthog=False)
#print('Solution: %r' % result.coef)

plt.clf()
plt.subplot(211)
plt.plot(domain, y)
plt.subplot(212)
plt.plot(domain, reconstruction)
plt.plot(domain, resid, c='red')
plt.show()

# generate a dictionary and manufactured solution
# X = np.array([np.cos(2 * np.pi * m * domain) for m in range(4)]).T
