import torch
import cooper
import einops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def make_symmetric(t: torch.Tensor) -> torch.Tensor:

    return (t + einops.rearrange(t, 'α β n -> β α n')) / 2.0


class Model(torch.nn.Module):

    num_types: int
    interaction_order: int
    adjacency_tensor: torch.Tensor

    def __init__(self, num_types, interaction_order, adjacency_tensor, interaction_tensor=None):

        super().__init__()
        self.num_types = num_types
        self.interaction_order = interaction_order
        self.adjacency_tensor = adjacency_tensor
        if interaction_tensor is None:
            interaction_tensor = torch.rand((self.num_types, self.num_types, self.interaction_order))
        self.interaction_tensor = torch.nn.Parameter(make_symmetric(interaction_tensor))

    def forward(self, v_batch):

        return 0.5 * einops.einsum(
            self.interaction_tensor,
            self.adjacency_tensor,
            v_batch,
            v_batch,
            'α β n, i j n, i α s, j β s -> s'
        )


class SymmetryConstraint(cooper.ConstrainedMinimizationProblem):

    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.criterion = torch.nn.MSELoss()
        super().__init__(is_constrained=True)

    def closure(self):

        # transpose matrices by permuting axes
        transposed = einops.rearrange(
            self.model.interaction_tensor,
            'α β n -> β α n'
        )
        square_diff = (self.model.interaction_tensor - transposed) ** 2

        return cooper.CMPState(
            loss=self.criterion(self.model.forward(self.inputs), self.targets),
            eq_defect=square_diff.flatten().sum()
        )


def widget(epoch, total_epochs, info: str = None):

    num_bars_total = 50
    proportion_finished = (epoch + 1) / total_epochs
    num_bars_filled = int(num_bars_total * proportion_finished)

    progress_bar = ''.join(['\\'] * num_bars_filled) + ''.join(['|'] * (num_bars_total - num_bars_filled))
    if proportion_finished == 1.0:
        end_char = '\n'
    else:
        end_char = ''
    if info:
        print(f'\r{progress_bar} {info} {proportion_finished * 100:.1f}%\r', end=end_char)
    else:
        print(f'\r{progress_bar} {proportion_finished * 100:.1f}', end=end_char)


def main():

    num_types = 3
    interaction_order = 2
    num_sites = 200
    num_samples = 50
    num_epochs = 100_000

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    all_occupation_numbers = np.zeros((num_sites, num_types, num_samples))

    for sample in np.arange(num_samples, dtype=int):
        occupation_numbers_ = np.zeros((num_sites, num_types))
        for site in np.arange(num_sites, dtype=int):
            type_ = np.random.choice(np.arange(num_types, dtype=int))
            occupation_numbers_[site, type_] = 1.0
        all_occupation_numbers[:, :, sample] = occupation_numbers_

    occupation_numbers = torch.Tensor(all_occupation_numbers)
    energies = torch.rand(num_samples)
    adjacency_tensor = (torch.rand(size=(num_sites, num_sites, interaction_order)) < 0.05).float()

    model = Model(
        num_types=num_types,
        interaction_order=interaction_order,
        adjacency_tensor=adjacency_tensor
    ).to(device)

    cmp = SymmetryConstraint(model, occupation_numbers, energies)
    formulation = cooper.LagrangianFormulation(cmp)

    primal_optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
    dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=1e+7)

    coop = cooper.ConstrainedOptimizer(
        formulation=formulation,
        primal_optimizer=primal_optimizer,
        dual_optimizer=dual_optimizer,
    )

    epochs = np.arange(num_epochs, dtype=int)
    eq_defects = np.zeros(num_epochs)
    losses = np.zeros(num_epochs)

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

    u = make_symmetric(model.interaction_tensor)
    with torch.no_grad():
        new_model = Model(num_types, interaction_order, adjacency_tensor, u)
        predicted = cmp.criterion(model.forward(occupation_numbers), energies)
        new_predicted = cmp.criterion(new_model.forward(occupation_numbers), energies)

    print(f'Close to symmetric loss: {predicted:.2E}')
    print(f'Symmetrized loss: {new_predicted:.2E}')

    mpl.use('Agg')
    plt.plot(epochs / 10_000, eq_defects, label='defect')
    plt.plot(epochs / 10_000, losses, label='loss')
    plt.grid()
    plt.xlabel(r'epoch / $10^4$')
    plt.ylabel('defect/loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig('optimization.svg', bbox_inches='tight')


if __name__ == '__main__':

    main()
