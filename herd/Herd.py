from itertools import product, combinations, starmap
from functools import partial
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import KernelDensity as KD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


@dataclass
class CMD:
    name: str
    cluster: Any
    pipe: Any
    doublepipe: Any
    sscores: Any
    dscores: Any
    

def pca_kde_pipe():
    pca = PCA(n_components=2, whiten=True)
    kde = KD(kernel='gaussian')
    return Pipeline([('pca', pca), ('kde', kde)])


def samps_vstack(colors, mags):
    return np.vstack([np.array(x) for x in [colors, mags]]).T


def pca_kde_fit(pipe, colors, mags):
    X = samps_vstack(colors, mags)
    params = dict(kde__bandwidth=np.logspace(-2, 2, 200))
    grid = GridSearchCV(pipe, params, cv=max(11, X.shape[0])-1)
    grid.fit(X)
    newpipe = grid.best_estimator_.fit(X)
    return newpipe


def gen_grid(xlims, ylims, ranges):
    xmin, xmax = xlims
    ymin, ymax = ylims
    xlen, ylen = ranges
    xspace = np.linspace(xmin, xmax, xlen)
    yspace = np.linspace(ymin, ymax, ylen)
    X, Y = np.mgrid[xmin:xmax:xlen*1j, ymin:ymax:ylen*1j]
    return xspace, yspace, X, Y


def pca_kde_score(newpipe, grid):
    xspace, yspace = grid
    eval_where = np.array(list(product(xspace, yspace)))
    log_dens = newpipe[1].score_samples(eval_where)
    return log_dens.reshape(len(xspace), len(yspace))


def kde_ipca(newpipe, grids, lens):
    xlen, ylen = lens
    A, B = grids
    C_grid = np.vstack([np.ravel(A), np.ravel(B)]).T
    E_grid = newpipe[0].inverse_transform(C_grid)[:, 0].reshape(xlen, ylen)
    F_grid = newpipe[0].inverse_transform(C_grid)[:, 1].reshape(xlen, ylen)
    return E_grid, F_grid


def get_cmsamps(newpipe, nsamps):
    samps = np.array([[x, y] for x, y in newpipe[1].sample(nsamps)])
    cmsamps = newpipe[0].inverse_transform(samps)
    return (*(cmsamps[:, i] for i in [0, 1]),)


def get_added_samps(newpipe, nsamps, multi=2):
    colors, mags = get_cmsamps(newpipe, nsamps)
    nwiseargs = [multi, mags, colors]
    return add_helper(magsum, *nwiseargs), add_helper(colorsum, *nwiseargs)


def add_helper(f, *nwiseargs):
    return np.fromiter(starmap(f, nwise(*nwiseargs)), np.float64)


def toflux(m):
    return 10.**(-2/5.*m)


def tomag(φ):
    return -5./2*np.log10(ϕ)


def magsum(m, c):
    return tomag(sum(((toflux(g) for g in m))))


def colorsum(m, c):
    return tomag(sum((toflux(g+h) for g, h in zip(m, c))))-magsum(m, c)


def nwise(n, *xy):
    x, y = xy
    return (map(np.array, zip(*a)) for a in combinations(zip(x, y), n))


def get_scores(colors, mags, apipe):
    X = samps_vstack(colors, mags)
    X_transform = apipe[0].transform(X)
    return apipe[-1].score_samples(X_transform)


def end2end(cluster, verbose=True):
    name = cluster['Cluster'][0]
    if verbose: print("Starting on cluster: {}".format(name))
    the_color = 'BP-RP'
    the_mag = 'phot_rp_mean_mag'
    colors, mags = cluster[the_color], cluster[the_mag]
    # xmin, xmax = (-4., 4.)
    # ymin, ymax = (-4., 4.)
    # xlen, ylen = (50, 50)
    # xlims = xmin, xmax
    # ylims = ymin, ymax
    # ranges = xlen, ylen
    pipe = pca_kde_pipe()
    if verbose: print("Starting fit")
    pipe = pca_kde_fit(pipe, colors, mags)
    # xlin, ylin, xgrid, ygrid = gen_grid(xlims, ylims, ranges)
    # grids = xgrid, ygrid
    if verbose: print("Adding samples")
    newmags, newcolors = get_added_samps(pipe, 100)
    if verbose: print("Fitting doublepipe")
    doublepipe = pca_kde_fit(pca_kde_pipe(),
                             np.random.choice(newcolors, 100),
                             np.random.choice(newmags, 100))
    if verbose: print("Scoring samples")
    wget_scores = partial(get_scores, colors, mags)
    sscores = wget_scores(pipe)
    dscores = wget_scores(doublepipe)
    # dcolors, dmags = colors[sscores < dscores], mags[sscores < dscores]
    # newnewpipe = pca_kde_fit(pca_kde_pipe(), dcolors, dmags)
    # Make this into a named tuple, probably
    return CMD(name, cluster, pipe, doublepipe, sscores, dscores)
