#!/usr/bin/python

from cluster_expand import ClusterExpansion
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():

    num_types = 3
    interaction_order = 2
    num_sites = 200
    num_samples = 50
    num_epochs = 10_000

    # initialize occupation numbers, where each site can only be occupied by a single type
    all_occupation_numbers = np.zeros((num_sites, num_types, num_samples))
    for sample in np.arange(num_samples, dtype=int):
        occupation_numbers_ = np.zeros((num_sites, num_types))
        for site in np.arange(num_sites, dtype=int):
            type_ = np.random.choice(np.arange(num_types, dtype=int))
            occupation_numbers_[site, type_] = 1.0
        all_occupation_numbers[:, :, sample] = occupation_numbers_

    # randomly initialize energies and a sparse adjacency tensor with 5% 1's
    energies = np.random.uniform(low=-5.0, high=-2.0, size=num_samples)
    adjacency_tensor = np.random.choice([0, 1], size=(num_sites, num_sites, interaction_order), p=[0.95, 0.05])

    expansion = ClusterExpansion(
        all_occupation_numbers,
        energies,
        adjacency_tensor,
        interaction_order,
        num_epochs
    )

    eq_defects, losses, interaction_matrix = expansion.fit()

    mpl.use('Agg')
    plt.plot(eq_defects, label='defect')
    plt.plot(losses, label='loss')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.savefig('optimization.svg')


if __name__ == '__main__':

    main()
