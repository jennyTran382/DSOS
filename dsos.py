__author__ = 'Su Nguyen and Binh Tran'
import numpy as np
import numpy.matlib
from tqdm import tqdm
from tqdm import trange


def distance(x, y):
    return np.linalg.norm((y - x))

# def learning(data=None, nepoch=None, spread_factor=0.1, age_max=None, lamb=None, c1=None, c2=None):
def learning(input_data=None, embedding_dimension=2, max_nepoch=None, spread_factor=0.1, lamb=None, alpha=0.1):
    """ A fast ATL function """
    resol = spread_factor
    trace_energy = []
    trace_error = []
    # Initialize 2 nodes
    data = np.copy(input_data)
    np.random.shuffle(data)
    particle = np.copy(data[0:2, :])
    # Binh added
    par_exp = np.copy(particle)
    energy = np.array([0.5, 0.5])  # maximum energy per particle is 1.0; if zeros --> hypernate mode
    Z = None
    m = np.array([1, 1])
    tem = distance(data[0, :], data[1, :])
    threshold = np.array([tem, tem])
    connections = np.array([[0, 0], [0, 0]])
    repulsive_force = np.array([[0.0, 0.0], [0.0, 0.0]])
    value = np.array([0., 0.], dtype=np.float64)
    index = np.array([0, 0], dtype=np.int32)
    # sample_size = data.shape[0]
    sample_size = data.shape[0] * max_nepoch
    max_repulsive_force = sample_size
    total_newnparticles = 0
    n_newparticles = 0
    for ii in trange(2, sample_size, desc='\tDSOS Progress', leave=True):
        # find winner node and runner-up node
        i = ii % data.shape[0]
        if i == 0:
            np.random.shuffle(data)

        dist = np.sqrt(np.sum(np.square(np.matlib.repmat(data[i, :], particle.shape[0], 1) - particle), axis=1))
        norm_dist = np.interp(dist, (dist.min(), dist.max()), (0, +1))

        value[0] = np.amin(dist)
        index[0] = np.argmin(dist)
        # if index[0] != 0:
        #     print()
        trace_error.append(dist[index[0]]*dist[index[0]])
        dist[index[0]] = 1000000
        value[1] = np.amin(dist)
        index[1] = np.argmin(dist)

        eps = 0.00001
        # prototype, connection and age update
        if value[0] > threshold[index[0]]/(eps+energy[index[0]]) or value[1] > threshold[index[1]]/(eps+energy[index[1]]):
        # if value[0] > threshold[index[0]] or value[1] > threshold[index[1]]:
            # add a new prototype
            # Binh added
            par_exp = np.concatenate((par_exp, np.reshape(data[i, :], (1, -1))), axis=0)
            particle = np.concatenate((particle, np.reshape(data[i, :], (1, -1))), axis=0)
            energy = np.concatenate((energy, np.array([0.5])))
            # Z = np.concatenate((Z, np.random.rand(1, embedding_dimension)), axis=0)
            threshold = np.concatenate((threshold, np.array([1000000])))
            m = np.concatenate((m, np.array([1])))
            connections = np.concatenate((connections, np.zeros((1, connections.shape[1]))), axis=0)
            connections = np.concatenate((connections, np.zeros((connections.shape[0], 1))), axis=1)
            repulsive_force = np.concatenate((repulsive_force, np.zeros((1, repulsive_force.shape[1]))), axis=0)
            repulsive_force = np.concatenate((repulsive_force, np.zeros((repulsive_force.shape[0], 1))), axis=1)
            n_newparticles += 1
        else:
            # find neighbor nodes of winner nodes
            neighbors = np.nonzero(connections[index[0], :])[0]
            if neighbors.shape[0] > 0:
                meandist = np.mean(dist[neighbors])
                nneighbors = np.sum(connections, axis=1)
                sum_energy = (energy[index[0]] + energy[neighbors])
                repulsive_force[index[0], neighbors] += 1 + sum_energy * nneighbors[neighbors] * dist[neighbors] / meandist
                repulsive_force[neighbors, index[0]] += 1 + sum_energy * nneighbors[neighbors] * dist[neighbors] / meandist
            # build connection
            connections[index[0], index[1]] = 1
            connections[index[1], index[0]] = 1
            repulsive_force[index[1], index[0]] = 0.0
            repulsive_force[index[0], index[1]] = 0.0
            neighbors = np.nonzero(connections[index[0], :])[0]
            # adjust the weight of winner node
            m[index[0]] += 1  # number of matches
            # particle[index[0], :] += (1.0 / np.float64(m[index[0]])) * (data[i, :] - particle[index[0], :])
            # Binh added to update best matched particle
            moving_step = 1 / np.float64(m[index[0]])
            particle[index[0], :] += (1 - alpha) * moving_step * (data[i, :] - particle[index[0], :]) + \
                                     alpha * moving_step * (par_exp[index[0], :] - particle[index[0], :])
            par_exp[index[0], :] = np.copy(data[i, :])
            energy[index[0]] += 1/(1+np.sum(norm_dist < resol))*(1.0 - norm_dist[index[0]])

            if neighbors.shape[0] > 0:
                # particle[neighbors, :] += (1.0 / (100.0 * np.float64(m[index[0]]))) * \
                #                            (np.matlib.repmat(data[i, :], neighbors.shape[0], 1) - particle[neighbors, :])
                # Binh added to update neighbours
                moving_step = 1 * np.reshape(((1-energy[neighbors]) / (100.0 * m[neighbors])) , (neighbors.shape[0], 1))
                particle[neighbors, :] += (1.0 - alpha) * moving_step * \
                                        (np.matlib.repmat(data[i, :], neighbors.shape[0], 1) - particle[neighbors, :]) + \
                                        alpha * moving_step * (par_exp[neighbors, :] - particle[neighbors, :])
                par_exp[neighbors, :] = np.matlib.repmat(data[i, :], neighbors.shape[0], 1)
                energy[neighbors] += 1.0/(1+np.sum(norm_dist < resol))*(1.0 - norm_dist[neighbors]) #neighbors1.shape[0]+1
            # delete the edges whose ages are greater than age_ma
            locate = np.where(repulsive_force[index[0], :] > max_repulsive_force)[0]
            connections[index[0], locate] = 0
            connections[locate, index[0]] = 0
            repulsive_force[index[0], locate] = 0
            repulsive_force[locate, index[0]] = 0

        # update threshold
        if np.count_nonzero(connections[index[0], :]) == 0:
            # no neighbor, the threshold should be the distance between winner node and runner-up node
            threshold[index[0]] = distance(particle[index[0], :], particle[index[1], :])
        else:
            # if have neighbors1, choose the farthest one
            neighbors = np.nonzero(connections[index[0], :])[0]
            neighbor_distances = np.matlib.repmat(particle[index[0], :], neighbors.shape[0], 1) - particle[neighbors,
                                                                                                   :]
            threshold_winner = np.max(np.sqrt(np.sum(np.square(neighbor_distances), axis=1)))
            threshold[index[0]] = threshold_winner

        if np.count_nonzero(connections[index[1], :]) == 0:
            # no neighbor
            threshold[index[1]] = distance(particle[index[0], :], particle[index[1], :])
        else:
            neighbors = np.nonzero(connections[index[1], :])[0]
            neighbor_distances = np.matlib.repmat(particle[index[1], :], neighbors.shape[0], 1) - particle[neighbors,
                                                                                                   :]
            threshold_runner = np.max(np.sqrt(np.sum(np.square(neighbor_distances), axis=1)))
            threshold[index[1]] = threshold_runner
        # print(i, "/", sample_size)

        # update step energy spent
        energy -= 1.0 / lamb #particle.shape[0] #0.1 / (lamb)  # constant spent energy (per step) -- if nothing particle is not hit in a lambda cycle, its energy will be drained out.
        energy = np.clip(energy, 0, 1.0)
        trace_energy.append([np.sum(energy), np.mean(energy), particle.shape[0]])
        if (ii + 1) % lamb == 0 or (ii == sample_size - 1):  # or :
            # delete nodes with 0, 1 or 2 neighbors1
            degrees = np.sum(connections, axis=1)
            neighbor0_set = np.where(degrees == 0)[0]
            neighbor1_set = np.where(degrees == 1)[0] #np.intersect1d(np.where(energy==0)[0], np.where(neighbors1 == 1)[0]) ##
            neighbor2_set = np.intersect1d(np.where(energy==0)[0],
                                           np.where(degrees == 2)[0])
            if ii == sample_size - 1:
                # to_delete = np.array([])
                # to_delete = neighbor0_set
                to_delete = np.union1d(neighbor0_set, neighbor1_set)
                # to_delete = np.union1d(to_delete, neighbor2_set)
            else:
                to_delete = np.union1d(neighbor0_set, neighbor1_set)
                to_delete = np.union1d(to_delete, neighbor2_set)
            if particle.shape[0] - to_delete.shape[0] < 2:
                continue
            particle = np.delete(particle, to_delete, axis=0)
            energy = np.delete(energy, to_delete, axis=0)
            threshold = np.delete(threshold, to_delete)
            m = np.delete(m, to_delete)
            connections = np.delete(connections, to_delete, axis=0)
            connections = np.delete(connections, to_delete, axis=1)
            repulsive_force = np.delete(repulsive_force, to_delete, axis=0)
            repulsive_force = np.delete(repulsive_force, to_delete, axis=1)
            max_repulsive_force = np.sum(energy)/3 #(lamb + particle.shape[0]) / 2 * spread_factor
            total_newnparticles += n_newparticles
            n_newparticles = 0


    return particle, connections, Z, energy, trace_energy, trace_error
