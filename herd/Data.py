import numpy as np
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier


def get_vizier_catalog():
    catalog = Vizier(row_limit=20000).get_catalogs("J/A+A/618/A93")
    return catalog[1].group_by('Cluster')


def get_vizier_cluster(catalog, clusterid=41):
    cluster = catalog.groups[clusterid]
    return cluster


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
    catalog = get_vizier_catalog()
    vclus = get_vizier_cluster(catalog, clusterid)
    cutclus = cut_cluster(vclus)
    augclus = augment_with_gaia(cutclus)
    return augclus


def gen_cluster():
    catalog = get_vizier_catalog()
    for cluster in catalog.groups:
        yield augment_with_gaia(cut_cluster(cluster))
