# David Mykytyn and Alex Malz
from itertools import product, combinations, count, permutations, starmap, chain, repeat
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import astropy as ap
import sklearn
from sklearn.neighbors import KernelDensity as KD
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
import corner
from sklearn.mixture import GaussianMixture as GM
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.pipeline import Pipeline
from scipy.stats import norm
from matplotlib.colors import LogNorm
from astroquery.gaia import Gaia
from scipy import stats
import seaborn as sns


def get_vizier_cluster(clusterid=41):
    Vizierinit = Vizier(row_limit=20000)
    catalog = Vizierinit.get_catalogs("J/A+A/618/A93")
    clucata = catalog[1]
    newc = clucata.group_by('Cluster')
    maxcluster = np.argmax(newc.groups.indices[1:]-newc.groups.indices[:-1])
    bigcluster = newc.groups[clusterid]
    return bigcluster

def cut_cluster(cluster):
    # Good `'Cluster'`s to choose from: Alessi\_24, ASCC_99, Alessi\_12(=41)
    cutcluster = cluster[cluster['PMemb']>.8]
    cutcluster = cutcluster[~np.isnan(cutcluster["BP-RP"])]
    #We should cut by lines away from main sequence
    cutcluster = cutcluster[~np.logical_and(cutcluster["BP-RP"]>1.0,cutcluster['Gmag']<10.)]
    cutcluster = cutcluster[~np.logical_and(cutcluster["BP-RP"]<.7, cutcluster['Gmag']>13.8)]
    return cutcluster

def augment_with_gaia(cluster):
    query = 'SELECT source_id, phot_bp_mean_mag, phot_rp_mean_mag FROM gaiadr2.gaia_source WHERE source_id = {}'
    newquery = query.format(' OR source_id = '.join(list(map(str, cluster['Source']))))
    newjob = Gaia.launch_job(query=newquery)
    results = newjob.get_results()
    results['phot_bp_mean_mag'].info.parent_table;
    cluster.add_columns([results['phot_bp_mean_mag'],results['phot_rp_mean_mag']])
    return cluster

def get_cluster(clusterid=41):
    return augment_with_gaia(cut_cluster(get_vizier_cluster(clusterid)))


def pca_kde_pipe():
    pca = PCA(n_components=2, whiten=True)
    kde = KD(kernel='gaussian')
    return Pipeline([('pca',pca),('kde',kde)])

def pca_kde_fit(pipe, colors, mags, xlen, ylen):
    X = np.vstack([np.array(x) for x in [colors, mags]]).T
    X_pipe = pipe.fit(X)
    params = dict(kde__bandwidth=np.logspace(-2, 2, 200))
    grid = GridSearchCV(pipe, params, cv=10)
    grid.fit(X)
    newpipe = grid.best_estimator_.fit(X)
    return newpipe

def pca_kde_score(newpipe, xlims, ylims, ranges):
    xmin, xmax = xlims
    ymin, ymax = ylims
    xlen, ylen = ranges
    eval_where = np.array(list(product(np.linspace(xmin,xmax,xlen), np.linspace(ymin,ymax, ylen))))
    A, B = np.mgrid[xmin:xmax:xlen*1j,ymin:ymax:ylen*1j]
    log_dens = newpipe[1].score_samples(eval_where)
    return (A, B, log_dens.reshape(xlen, ylen))
#xmin, xmax = (-4., 4.)
#ymin, ymax = (-4., 4.)
#xlen, ylen = (50, 50)
#testpipe = pca_kde_fit(cutcluster[the_color], cutcluster[the_mag], 50, 50)
#xgrid, ygrid, logdens = pca_kde_score(testpipe, (-4., 4.), (-4., 4.), (50, 50))
#plt.contour(xgrid, ygrid, np.exp(logdens.reshape(xlen,ylen)));
def kde_ipca(newpipe, grids, lens):
    xlen, ylen = lens
    A, B = grids
    C_grid = np.vstack([np.ravel(A), np.ravel(B)]).T
    E_grid = newpipe[0].inverse_transform(C_grid)[:,0].reshape(xlen, ylen)
    F_grid = newpipe[0].inverse_transform(C_grid)[:,1].reshape(xlen, ylen)
    return E_grid, F_grid
#α, β = kde_ipca(testpipe, (xgrid, ygrid), (50, 50))
def get_cmsamps(newpipe, nsamps):
    samps = np.array([[x,y] for x,y in newpipe[1].sample(nsamps)])
    cmsamps = newpipe[0].inverse_transform(samps)
    return (*(cmsamps[:,i] for i in [0,1]),)
def get_added_samps(newpipe,nsamps,multi=2):
    colors, mags = get_cmsamps(newpipe, nsamps)
    return [np.fromiter(starmap(z, nwise(multi)(mags, colors)), np.float64) for z in [magsum, colorsum]]
toflux, tomag = lambda m: 10.**(-2/5.*m), lambda ϕ: -5./2*np.log10(ϕ)
magsum = lambda m, c: tomag(sum(((toflux(g) for g in m))))
colorsum = lambda m, c: tomag(sum((toflux(g+h) for g,h in zip(m,c))))-magsum(m,c)
nwise = lambda n: lambda x,y: (map(np.array, zip(*a)) for a in combinations(zip(x,y),n))
#newmags, newcolors = get_added_samps(testpipe, 100)
#doublepipe = pca_kde_fit(newcolors, newmags, 50, 50)
#xgrid, ygrid, logdens = pca_kde_score(doublepipe, (-4., 4.), (-4., 4.), (50, 50))
#sns.kdeplot(newcolors, newmags, label='KDE(pairwise added samples)',alpha=0.5)
#plt.plot(cutcluster[the_color],cutcluster[the_mag], 'r+',alpha=.4, label='original data')
#plt.legend()
#plt.ylim(18,7)
# ### END OF REAL WORK
