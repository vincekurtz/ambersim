import jax
import jax.numpy as jp

from ambersim.reinforcement_learning.policies import (
    MLP,
    HierarchyComposition,
    ParallelComposition,
    SequentialComposition,
)


def test_mlp():
    """Test creating and evaluating a basic MLP."""
    input_size = (3,)
    layer_sizes = (2, 3, 4)

    # Create the MLP
    rng = jax.random.PRNGKey(0)
    mlp = MLP(layer_sizes=layer_sizes, bias=True)
    dummy_input = jp.ones(input_size)
    params = mlp.init(rng, dummy_input)

    # Check the MLP's structure
    print(mlp.tabulate(rng, dummy_input))

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = mlp.apply(params, my_input)
    assert my_output.shape[-1] == layer_sizes[-1]

    # Check number of parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    expected_num_params = 0
    sizes = input_size + layer_sizes
    for i in range(len(sizes) - 1):
        expected_num_params += sizes[i] * sizes[i + 1]  # weights
        expected_num_params += sizes[i + 1]  # biases
    assert num_params == expected_num_params


def test_sequential():
    """Test creating and evaluating a sequential composition of modules."""
    # Each module is an MLP with these dimensions
    input_size = (2,)
    hidden_sizes = (128,) * 2
    output_size = (3,)

    # Create the sequential composition
    num_modules = 3
    rng = jax.random.PRNGKey(0)
    net = SequentialComposition(MLP, num_modules, module_kwargs={"layer_sizes": hidden_sizes + output_size})
    dummy_input = jp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Check the MLP's structure
    print(net.tabulate(rng, dummy_input, depth=1))

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)
    assert my_output.shape[-1] == output_size[-1]


def test_parallel():
    """Test creating and evaluating a parallel composition of modules."""
    # Each module is an MLP with these dimensions
    input_size = (2,)
    hidden_sizes = (128,) * 2
    output_size = (3,)

    # Create the parallel composition
    num_modules = 3
    rng = jax.random.PRNGKey(0)
    net = ParallelComposition(MLP, num_modules, module_kwargs={"layer_sizes": hidden_sizes + output_size})
    dummy_input = jp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Check the MLP's structure
    print(net.tabulate(rng, dummy_input, depth=1))

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)
    assert my_output.shape[-1] == output_size[-1]


def test_hierarchy():
    """Test creating and evaluating a hierarchical composition of modules."""
    # Each module is an MLP with these dimensions
    input_size = (2,)
    hidden_sizes = (128,) * 2
    output_size = (3,)

    # Create the parallel composition
    num_modules = 3
    rng = jax.random.PRNGKey(0)
    net = HierarchyComposition(MLP, num_modules, module_kwargs={"layer_sizes": hidden_sizes + output_size})
    dummy_input = jp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Check the MLP's structure
    print(net.tabulate(rng, dummy_input, depth=1))

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)
    assert my_output.shape[-1] == output_size[-1]
