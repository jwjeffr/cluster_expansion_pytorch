.. role:: math(raw)
    :format: latex html

.. _cooper: https://github.com/cooper-org/cooper
.. _einops: https://github.com/arogozhnikov/einops

Cluster Expansion Fitting Using PyTorch
#######################################

The goal of the cluster expansion is to fit energy data to some Hamiltonian (using the Einstein summation convention):

:math:`$$\mathcal{H} = N_{\alpha\beta n}u_{\alpha\beta n}$$`

where :math:`$N_{\alpha\beta n}$` is the number of :math:`$\alpha$`-:math:`$\beta$` bonds in the :math:`$n$`'th neighbor shell and :math:`$u_{\alpha\beta n}$` is the :math:`$\alpha$`-:math:`$\beta$` interaction at the :math:`$n$`'th nearest neighbor distance. Note that Greek letters index atom types.

We can count the number of bonds by occupation matrices :math:`$x_{i\alpha}$`, where:

:math:`$$x_{i\alpha} = \begin{cases} 1 & \text{site $i$ occupied by $\alpha$} \\ 0 & \text{else} \end{cases}$$`

Then, the Hamiltonian is:

:math:`$$\mathcal{H} = \frac{1}{2}u_{\alpha\beta n}A_{ijn} x_{i\alpha} x_{j\beta}$$`

where :math:`$A$` is defined by:

:math:`$$A_{ijn} = \begin{cases} 1 & \text{sites $i$ and $j$ are $n$'th nearest neighbors} \\ 0 & \text{else} \end{cases}$$`

Note that :math:`$u_{\alpha\beta n}$` is intensive, and therefore can be used on a larger lattice with a larger adjacency tensor.

This repository contains a library (``cluster_expand.py``) and an example using the library (``example.py``) that uses PyTorch, `einops`_, and `cooper`_ to fit :math:`$u_{\alpha\beta n}$` in terms of configuration matrices and energies with an input adjacency tensor.

- PyTorch is used to create a Model class and accelerate the tensor operations and optimization
- Einops is used to simplify the syntax for the tensor operations such as Einstein summations
- Cooper is used to constrain the optimization such that :math:`$u_{\alpha\beta n} = u_{\beta\alpha n}$`

