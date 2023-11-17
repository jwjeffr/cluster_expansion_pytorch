#!/usr/bin/python

import torch
import cooper
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def make_symmetric(t: torch.Tensor) -> torch.Tensor:

    """
    Function to make a tensor symmetric about its first two axes
    :param t: tensor
    :return: symmetrized tensor
    """

    return (t + einops.rearrange(t, 'α β n -> β α n')) / 2.0


class Model(torch.nn.Module):

    """
    Class for creating model to optimize
    """

    num_types: int
    interaction_order: int
    adjacency_tensor: torch.Tensor

    def __init__(self, num_types, interaction_order, adjacency_tensor, interaction_tensor=None):

        super().__init__()
        self.num_types = num_types
        self.interaction_order = interaction_order
        self.adjacency_tensor = adjacency_tensor

        # if interaction tensor not specified, create a random one
        if interaction_tensor is None:
            interaction_tensor = torch.rand((self.num_types, self.num_types, self.interaction_order))

        # symmetrize whatever interaction tensor is given, store as a model parameter
        self.interaction_tensor = torch.nn.Parameter(make_symmetric(interaction_tensor))

    def forward(self, x_batch):

        """
        Forward pass function in terms of x_batch
        :param x_batch: (num_sites, num_types, num_samples) size tensor, where each (num_sites, num_types) slice
        is the configuration matrix for the sample
        :return: energies of each sample indexed by s
        """

        return 0.5 * einops.einsum(
            self.interaction_tensor,
            self.adjacency_tensor,
            x_batch,
            x_batch,
            'α β n, i j n, i α s, j β s -> s'
        )


class SymmetryConstraint(cooper.ConstrainedMinimizationProblem):

    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = targets

        # initialize loss as mean square error
        self.criterion = torch.nn.MSELoss()
        super().__init__(is_constrained=True)

    def closure(self):

        """
        Closure method that defines constraint
        :return: CMPState
        """

        # transpose matrices by permuting axes
        transposed = einops.rearrange(self.model.interaction_tensor, 'α β n -> β α n')

        # compute deviation from symmetry
        square_diff = (self.model.interaction_tensor - transposed) ** 2

        # return constraint object
        return cooper.CMPState(
            loss=self.criterion(self.model.forward(self.inputs), self.targets),
            eq_defect=square_diff.flatten().sum()
        )


def widget(epoch, total_epochs, info: str = None):

    """
    Widget showing progress of training
    :param epoch: current training epoch
    :param total_epochs: total epoch
    :param info: optional string to print out after progress bar
    :return: None
    """

    num_bars_total = 50
    proportion_finished = (epoch + 1) / total_epochs
    num_bars_filled = int(num_bars_total * proportion_finished)

    progress_bar = ''.join(['\\'] * num_bars_filled) + ''.join(['|'] * (num_bars_total - num_bars_filled))
    if proportion_finished == 1.0:
        end_char = '\n'
    else:
        end_char = ''
    if info:
        print(f'\r{progress_bar} {proportion_finished * 100:.1f}% {info}\r', end=end_char)
    else:
        print(f'\r{progress_bar} {proportion_finished * 100:.1f}', end=end_char)


def main():

    # define some constants, interaction_order is the number of shells to care about
    num_types = 3
    interaction_order = 2
    num_sites = 200
    num_samples = 50
    num_epochs = 1_000_000

    # use cuda if torch can find it
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # initialize occupation numbers, where each site can only be occupied by a single type
    all_occupation_numbers = np.zeros((num_sites, num_types, num_samples))
    for sample in np.arange(num_samples, dtype=int):
        occupation_numbers_ = np.zeros((num_sites, num_types))
        for site in np.arange(num_sites, dtype=int):
            type_ = np.random.choice(np.arange(num_types, dtype=int))
            occupation_numbers_[site, type_] = 1.0
        all_occupation_numbers[:, :, sample] = occupation_numbers_

    # turn occupation numbers into a tensor
    occupation_numbers = torch.Tensor(all_occupation_numbers)

    # randomly initialize energies and a sparse adjacency tensor
    # IMPORTANT: replace this with your actual data
    energies = torch.rand(num_samples)
    adjacency_tensor = (torch.rand(size=(num_sites, num_sites, interaction_order)) < 0.05).float()

    # initialize model, send it to device
    model = Model(
        num_types=num_types,
        interaction_order=interaction_order,
        adjacency_tensor=adjacency_tensor
    ).to(device)

    # create the Lagrangian for the constraint, initialize optimizers, one for objective function and one for constraint
    cmp = SymmetryConstraint(model, occupation_numbers, energies)
    formulation = cooper.LagrangianFormulation(cmp)
    primal_optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e+7)

    # initialize constrained optimizer
    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )

    epochs = np.arange(num_epochs, dtype=int)
    eq_defects = np.zeros(num_epochs)
    losses = np.zeros(num_epochs)

    # train
    for epoch in epochs:

        coop.zero_grad()
        lagrangian = formulation.composite_objective(cmp.closure)
        formulation.custom_backward(lagrangian)
        coop.step(cmp.closure)

        defect, loss = cmp.state.eq_defect, cmp.state.loss

        info = f'constraint loss: {defect:.2E}, overall loss: {loss:.2E}'
        widget(epoch, num_epochs, info)

        eq_defects[epoch] = defect
        losses[epoch] = loss

    # make resulting tensor fully symmetric instead of close-to-symmetric, evaluate loss with this last step
    u = make_symmetric(model.interaction_tensor)
    with torch.no_grad():
        new_model = Model(num_types, interaction_order, adjacency_tensor, u)
        predicted = cmp.criterion(model.forward(occupation_numbers), energies)
        new_predicted = cmp.criterion(new_model.forward(occupation_numbers), energies)

    print(f'Close to symmetric loss: {predicted:.2E}')
    print(f'Symmetrized loss: {new_predicted:.2E}')

    mpl.use('Agg')

    # plot training results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epochs / 10_000, eq_defects, label='defect')
    ax.plot(epochs / 10_000, losses, label='loss')
    ax.grid()
    ax.set_xlabel(r'epoch / $10^4$')
    ax.set_ylabel('defect/loss')
    ax.legend()
    ax.set_yscale('log')
    fig.savefig('optimization.svg', bbox_inches='tight')

    # plot interaction tensor
    fig, axs = plt.subplots(ncols=interaction_order)
    for i, ax in enumerate(axs):
        print(f"order {i + 1} interaction matrix:")
        print(u[:, :, i].detach().cpu().numpy())


if __name__ == '__main__':

    main()
