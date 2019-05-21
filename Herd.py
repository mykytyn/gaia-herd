from itertools import product, combinations, starmap

import numpy as np
from astroquery.vizier import Vizier
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity as KD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from astroquery.gaia import Gaia


def get_vizier_cluster(clusterid=41):
    Vizierinit = Vizier(row_limit=20000)
    catalog = Vizierinit.get_catalogs("J/A+A/618/A93")
    clucata = catalog[1]
    newc = clucata.group_by('Cluster')
    bigcluster = newc.groups[clusterid]
    return bigcluster


def cut_cluster(cluster):
    # Good `'Cluster'`s to choose from: Alessi\_24, ASCC_99, Alessi\_12(=41)
    cutcluster = cluster[cluster['PMemb'] > .8]
    cutcluster = cutcluster[~np.isnan(cutcluster["BP-RP"])]
    # We should cut by lines away from main sequence
    cutcluster = cutcluster[~np.logical_and(cutcluster["BP-RP"] > 1.0,
                                            cutcluster['Gmag'] < 10.)]
    cutcluster = cutcluster[~np.logical_and(cutcluster["BP-RP"] < .7,
                                            cutcluster['Gmag'] > 13.8)]
    return cutcluster


def augment_with_gaia(cluster):
    new_cols = ['phot_bp_mean_mag', 'phot_rp_mean_mag']
    cols = 'source_id, ' + ', '.join(new_cols)
    query = 'SELECT ' + cols + ' FROM gaiadr2.gaia_source WHERE source_id = '
    query += ' OR source_id = '.join(list(map(str, cluster['Source'])))
    newjob = Gaia.launch_job(query=query)
    results = newjob.get_results()
    cluster.add_columns([results[col] for col in new_cols])
    return cluster


def get_cluster(clusterid=41):
    vclus = get_vizier_cluster(clusterid)
    cutclus = cut_cluster(vclus)
    augclus = augment_with_gaia(cutclus)
    return augclus


def pca_kde_pipe():
    pca = PCA(n_components=2, whiten=True)
    kde = KD(kernel='gaussian')
    return Pipeline([('pca', pca), ('kde', kde)])


def samps_vstack(colors, mags):
    return np.vstack([np.array(x) for x in [colors, mags]]).T


def pca_kde_fit(pipe, colors, mags):
    X = samps_vstack(colors, mags)
    params = dict(kde__bandwidth=np.logspace(-2, 2, 200))
    grid = GridSearchCV(pipe, params, cv=10)
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
