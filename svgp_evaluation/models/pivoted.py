import torch

def _pivoted_cholesky_init(
    train_inputs,
    kernel_matrix,
    max_length,
    epsilon=1e-6,
):
    r"""
    A pivoted cholesky initialization method for the inducing points,
    originally proposed in [burt2020svgp]_ with the algorithm itself coming from
    [chen2018dpp]_. Code is a PyTorch version from [chen2018dpp]_, copied from
    https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py.

    Args:
        train_inputs: training inputs (of shape n x d)
        kernel_matrix: kernel matrix on the training
            inputs
        max_length: number of inducing points to initialize
        epsilon: numerical jitter for stability.

    Returns:
        max_length x d tensor of the training inputs corresponding to the top
        max_length pivots of the training kernel matrix
    """
    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: ensure this works in batch mode, which it does not currently.
    NEG_INF = -(torch.tensor(float("inf")))
    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros(
        (max_length, item_size), device=kernel_matrix.device, dtype=kernel_matrix.dtype
    )
    di2s = kernel_matrix.diag()
    selected_items = []
    selected_item = torch.argmax(di2s)
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = NEG_INF
        selected_item = torch.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    ind_points = train_inputs[torch.stack(selected_items)]

    return ind_points

def _select_inducing_points(
    inputs,
    covar_module,
    num_inducing,
    input_batch_shape,
):
    r"""
    Utility function that evaluates a kernel at given inputs and selects inducing point
    locations based on the pivoted Cholesky heuristic.

    Args:
        inputs: A (*batch_shape, n, d)-dim input data tensor.
        covar_module: GPyTorch Module returning a LazyTensor kernel matrix.
        num_inducing: The maximun number (m) of inducing points (m <= n).
        input_batch_shape: The non-task-related batch shape.

    Returns:
        A (*batch_shape, m, d)-dim tensor of inducing point locations.
    """

    train_train_kernel = covar_module(inputs).evaluate_kernel()

    # base case
    if train_train_kernel.ndimension() == 2:
        inducing_points = _pivoted_cholesky_init(
            train_inputs=inputs,
            kernel_matrix=train_train_kernel,
            max_length=num_inducing,
        )
    # multi-task case
    elif train_train_kernel.ndimension() == 3 and len(input_batch_shape) == 0:
        input_element = inputs[0] if inputs.ndimension() == 3 else inputs
        kernel_element = train_train_kernel[0]
        inducing_points = _pivoted_cholesky_init(
            train_inputs=input_element,
            kernel_matrix=kernel_element,
            max_length=num_inducing,
        )
    # batched input cases
    else:
        batched_inputs = (
            inputs.expand(*input_batch_shape, -1, -1)
            if inputs.ndimension() == 2
            else inputs
        )
        reshaped_inputs = batched_inputs.flatten(end_dim=-3)
        inducing_points = []
        for input_element in reshaped_inputs:
            # the extra kernel evals are a little wasteful but make it
            # easier to infer the task batch size
            kernel_element = covar_module(input_element).evaluate_kernel()
            # handle extra task batch dimension
            kernel_element = (
                kernel_element[0]
                if kernel_element.ndimension() == 3
                else kernel_element
            )
            inducing_points.append(
                _pivoted_cholesky_init(
                    train_inputs=input_element,
                    kernel_matrix=kernel_element,
                    max_length=num_inducing,
                )
            )
        inducing_points = torch.stack(inducing_points).view(
            *input_batch_shape, num_inducing, -1
        )

    return inducing_points
