import torch
import cooper
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass


def make_symmetric(t: torch.Tensor) -> torch.Tensor:

    """
    Function to make a tensor symmetric about its first two axes
    :param t: tensor
    :return: symmetrized tensor
    """

    return (t + torch.einsum('abn -> ban', t)) / 2.0


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

        return 0.5 * torch.einsum(
            'abn,ijn,ias,jbs -> s',
            self.interaction_tensor,
            self.adjacency_tensor,
            x_batch,
            x_batch
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
        transposed = torch.einsum('abn -> ban', self.model.interaction_tensor)

        # compute deviation from symmetry
        square_diff = (self.model.interaction_tensor - transposed) ** 2

        # return constraint object
        return cooper.CMPState(
            loss=self.criterion(self.model.forward(self.inputs), self.targets),
            eq_defect=square_diff.flatten().sum()
        )


@dataclass
class ClusterExpansion:

    """
    Class for cluster expansion fitting
    """

    occupation_tensor: ArrayLike
    energies: ArrayLike
    adjacency_tensor: ArrayLike
    interaction_order: int
    num_epochs: int
    primal_learning_rate: float = 1.0e-7
    dual_learning_rate: float = 1.0e+7

    def __post_init__(self):

        if type(self.occupation_tensor) is not torch.Tensor:
            self.occupation_tensor = torch.Tensor(self.occupation_tensor)
        if type(self.energies) is not torch.Tensor:
            self.energies = torch.Tensor(self.energies)
        if type(self.adjacency_tensor) is not torch.Tensor:
            self.adjacency_tensor = torch.Tensor(self.adjacency_tensor)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def widget(self, epoch, info: str = None):

        """
        Widget showing progress of training
        :param epoch: current training epoch
        :param info: optional string to print out after progress bar
        :return: None
        """

        num_bars_total = 50
        proportion_finished = (epoch + 1) / self.num_epochs
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

    def fit(self):

        num_sites, num_types, num_samples = self.occupation_tensor.shape

        model = Model(
            num_types=num_types,
            interaction_order=self.interaction_order,
            adjacency_tensor=self.adjacency_tensor
        ).to(self.device)

        # create the Lagrangian for the constraint
        # initialize optimizers, one for objective function and one for constraint
        cmp = SymmetryConstraint(model, self.occupation_tensor, self.energies)
        formulation = cooper.LagrangianFormulation(cmp)
        primal_optimizer = torch.optim.SGD(model.parameters(), lr=self.primal_learning_rate)
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=self.dual_learning_rate)

        # initialize constrained optimizer
        coop = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
        )

        epochs = np.arange(self.num_epochs, dtype=int)
        eq_defects = np.zeros(self.num_epochs)
        losses = np.zeros(self.num_epochs)

        # train
        for epoch in epochs:
            coop.zero_grad()
            lagrangian = formulation.composite_objective(cmp.closure)
            formulation.custom_backward(lagrangian)
            coop.step(cmp.closure)

            defect, loss = cmp.state.eq_defect, cmp.state.loss

            info = f'constraint loss: {defect:.2E}, overall loss: {loss:.2E}'
            self.widget(epoch, info)

            eq_defects[epoch] = defect
            losses[epoch] = loss

        # make resulting tensor fully symmetric instead of close-to-symmetric, evaluate loss with this last step
        u = make_symmetric(model.interaction_tensor)
        with torch.no_grad():
            new_model = Model(num_types, self.interaction_order, self.adjacency_tensor, u)
            predicted = cmp.criterion(model.forward(self.occupation_tensor), self.energies)
            new_predicted = cmp.criterion(new_model.forward(self.occupation_tensor), self.energies)

        u = u.detach().cpu().numpy()

        print(f'Close to symmetric loss: {predicted:.2E}')
        print(f'Symmetrized loss: {new_predicted:.2E}')

        return eq_defects, losses, u
