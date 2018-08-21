import numpy as np
import scipy.spatial.distance as dist
from scipy.special import binom


def init_update_dict_euclidean(f: np.ndarray, dict_faces: dict, toll: list, min_dist: list,  max_val: int):

    avg_old = 0.6
    avg_new = 0.4
    dist_limit = 1.15

    mean_tmp = 1.5

    # Initialization of dict_faces and toll first element
    if len(dict_faces) == 0:
        dict_faces[0] = [f]

        return dict_faces, 0

    for key in dict_faces.keys():
        br = False
        dist_f = np.zeros(len(dict_faces[key]))

        for val in range(0, len(dict_faces[key])):

            dist_f[val] = dist.euclidean(f, dict_faces[key][val])

            if dist_f[val] > dist_limit:
                br = True
                break
        if not br:

            mean_distance = np.average(dist_f)

            if mean_distance < toll[key] and mean_distance < mean_tmp:

                dist_f_tmp = dist_f
                key_tmp = key
                mean_tmp = mean_distance
                # significant value to add

    if mean_tmp == 1.5:

        new_id = key + 1
        dict_faces[new_id] = [f]
        return dict_faces, new_id

    min_distance = np.amin(dist_f_tmp)
    min_distance_id = np.argmin(dist_f_tmp)

    if min_distance > min_dist[key_tmp]:

        if len(dict_faces[key_tmp]) < max_val:
            dict_faces[key_tmp].insert(0, f)

        else:
            # update weighted average
            dict_faces[key_tmp][min_distance_id] = np.sum([avg_new * np.array(f), avg_old * np.array(dict_faces[key_tmp][min_distance_id])], axis=0)

    return dict_faces, key_tmp


def update_toll_euclidean(dict_faces: dict, key: int, toll: list, mean_dist: list, min_dist: list):

    max_toll = 1
    min_distance = 0.3

    avg_old_min = 0.8
    avg_new_min = 0.2

    w_maxtoll = 0.5
    w_toll = 0.3
    w_maxdist = 0.2

    if len(dict_faces.keys()) > len(toll):

        toll.insert(key, max_toll)
        min_dist.insert(key, min_distance)
        mean = (toll[key]+min_dist[key])/2
        mean_dist.insert(key, mean)

    elif len(dict_faces[key]) == 1:
        pass

    else:
        b = int(binom(len(dict_faces[key]), 2))
        dist_f = np.zeros(b)
        for f_k in range(0, len(dict_faces[key])):
            for f_j in range(f_k+1, len(dict_faces[key])):
                dist_f[f_k+f_j-1] = dist.euclidean(dict_faces[key][f_k], dict_faces[key][f_j])

        min_dist[key] = (np.amin(dist_f))*avg_new_min + min_distance * avg_old_min
        max_dist = np.amax(dist_f)

        if max_dist > mean_dist[key]:
            toll[key] = (toll[key]*w_toll)+(max_dist*w_maxdist)+(w_maxtoll*max_toll)
            mean_dist[key] = np.average(dist_f)

        else:
            mean_dist[key] = np.average(dist_f)

    return toll, mean_dist, min_dist


def bb_update_dict_euclidean(f: np.ndarray, dict_faces: dict, min_dist: list, max_val: int, l_key: int):
    avg_old = 0.7
    avg_new = 0.3
    dist_f = np.zeros(len(dict_faces[l_key]))

    for val in range(0, len(dict_faces[l_key])):
        dist_f[val] = dist.euclidean(f, dict_faces[l_key][val])

    min_distance = np.amin(dist_f)
    min_distance_id = np.argmin(dist_f)

    if min_distance > min_dist[l_key]:
        if len(dict_faces[l_key]) < max_val:
            dict_faces[l_key].insert(0, f)

        else:
            # update weighted average
            dict_faces[l_key][min_distance_id] = np.sum([avg_new * np.array(f),
                                                        avg_old * np.array(dict_faces[l_key][min_distance_id])], axis=0)

    return dict_faces, l_key


# --------------------------------------------------------------------------------------------------------------------



