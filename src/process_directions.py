import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from scipy import spatial
from matplotlib import pyplot as plt

def find_center(u):
    D = spatial.distance.squareform(spatial.distance.pdist(u, metric="euclidean"))
    return np.argmin(sum(D))

def get_cluster_idx(res, globla_idx):
    groups = []
    num_clusters = max(res.labels_) + 1
    for i in range(num_clusters):
        group_i = globla_idx[np.where(res.labels_ == i)[0]]
        groups.append(list(group_i))
    return groups

# read dataset and fitting results
def generate_directions(u, q=0.015, eps=None, num_cut=None):
    
    '''
        Given points u::tensor, m by d, detect clusters 
        and generate directions attached to cluster centers
    '''
    m, dim = u.shape
    pair_dist = spatial.distance.pdist(u, metric="euclidean")
    if eps is None:
        if num_cut is not None:
            eps = sorted(pair_dist)[num_cut]
        else:
            eps = np.quantile(pair_dist,q) 
    clustering = DBSCAN(eps=eps, min_samples=2).fit(u)
    print("eps = ", eps)
    num_clusters = max(clustering.labels_) + 1
    pairs = []
    for i in range(num_clusters):
        group_i = list(np.where(clustering.labels_ == i)[0])
        pairs.append(group_i)   

    new_pairs = []
    for group in pairs:
        if len(group) > dim+2:
            # subdivide 
            n_clusters = (len(group)//dim) - 1
            kmeans = KMeans(n_clusters=n_clusters).fit(u[group])
            group_small = get_cluster_idx(kmeans, np.array(group))
            new_pairs += group_small
        else:
            new_pairs.append(group)
    pairs = new_pairs
    num_clusters = len(new_pairs)

    V = dict()
    idx_to_remove = []
    num_directions = 0
    for i in range(num_clusters):
        idx_set = pairs[i]
        if len(idx_set) == 2:
            idx_center = idx_set[0]
            center_local_idx = 0
        else:
            center_local_idx = find_center(u[idx_set,:])
            idx_center = idx_set[center_local_idx]
        v_set = np.empty((0,dim))
        for j in range(len(idx_set)):
            if j != center_local_idx:
                idx_cur = idx_set[j]
                v = u[idx_cur,:] - u[idx_center,:]
                v /= np.linalg.norm(v)
                v_set = np.concatenate([v_set, v.reshape(1, -1)], axis=0)
                idx_to_remove.append(idx_cur)        
        V[idx_center] = v_set
        num_directions += v_set.shape[0]
        
    return V, idx_to_remove, num_directions


def plot_directions(u, V):
    # plot directions for sanity check 
    plt.figure(figsize=(10, 8))
    fig = plt.scatter(u[:,0], u[:,1])
    for key in V.keys():
        directions = V[key]
        print(f"cluster {key}, {directions.shape[0]} directions.")
        for i in range(directions.shape[0]):
            direction = directions[i]
            x1 = u[key]
            x2 = u[key] + direction/5
            fig = plt.plot([x1[0], x2[0]], [x1[1],x2[1]], c='#2ca02c')
        fig = plt.scatter(u[key,0], u[key,1], c="#ff7f0e")

    plt.savefig("./cluster_test.pdf")

def remove_directions(dd, angle_tol):
    # dd: m by d, m directions in dim d
    Dvec = spatial.distance.pdist(dd, metric="euclidean")
    Ddd = spatial.distance.squareform(Dvec)
    nd, dim = dd.shape
    Cdd = np.zeros((nd,nd))
    remove_idx = []
    for i in range(nd-1):
        if i not in remove_idx:
            for j in range(i+1,nd):
                if j not in remove_idx:
                    Cdd[i,j] = 2*np.arcsin(Ddd[i,j]/2)/(np.pi) * 180
                    if  Cdd[i,j] < angle_tol or Cdd[i,j] > (180-angle_tol):
                        remove_idx.append(j)
    return remove_idx, Cdd



def re_index(u, V, idx_to_remove):
    idx_to_keep = []
    for i in range(u.shape[0]):
        if i not in idx_to_remove:
            idx_to_keep.append(i)
    u2 = u[idx_to_keep]
    idx = np.array(range(u.shape[0]))
    idx2 = idx[idx_to_keep]
    V2 = dict()
    for i in range(u.shape[0]):
        if i in V.keys():
            new_key = np.where(idx2==i)[0][0]
            V2[new_key] = V[i]
    for key in V2.keys():
        remove_idx = remove_directions(V2[key], 10)[0]
        mask = np.ones(V2[key].shape[0], dtype=bool)
        mask[remove_idx] = False
        V2[key] = V2[key][mask]
    return u2, V2