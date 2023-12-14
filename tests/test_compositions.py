import jax
import jax.numpy as jp

from ambersim.learning.architectures import (
    MLP,
    HierarchyComposition,
    LinearSystemPolicy,
    NestedLinearPolicy,
    ParallelComposition,
    Quadratic,
    SeriesComposition,
    print_module_summary,
)


def test_quadratic():
    """Test creating and evaluating a quadratic module."""
    input_size = (2, 5)

    # Create the quadratic module
    rng = jax.random.PRNGKey(0)
    net = Quadratic(input_size[-1])
    dummy_input = jp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)

    print(my_input)
    print(my_output)

    assert my_output.shape[-1] == 1
    assert my_output.shape[0] == input_size[0]


def test_series():
    """Test creating and evaluating a series composition of modules."""
    # Each module is an MLP with these dimensions
    input_size = (2,)
    hidden_sizes = (128,) * 2
    output_size = (3,)

    # Create the sequential composition
    num_modules = 3
    rng = jax.random.PRNGKey(0)
    net = SeriesComposition(MLP, num_modules, module_kwargs={"layer_sizes": hidden_sizes + output_size})
    dummy_input = jp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Print the network summary
    print_module_summary(net, input_size)

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

    # Print the network summary
    print_module_summary(net, input_size)

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

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)
    assert my_output.shape[-1] == output_size[-1]


def test_nested_linear():
    """Test creating and evaluating a hierarchical of linear policies."""
    input_size = (2,)
    output_size = (3,)

    # Define the policy hyperparameters
    measurement_networks = [MLP, MLP, MLP]
    measurement_network_kwargs = [
        {"layer_sizes": (128, 128, 3)},
        {"layer_sizes": (128, 128, 3)},
        {"layer_sizes": (128, 128, 3)},
    ]
    linear_policy_kwargs = [
        {"features": 2, "use_bias": False},
        {"features": 2, "use_bias": False},
        {"features": output_size[0], "use_bias": False},
    ]

    # Create the nested linear policy
    rng = jax.random.PRNGKey(0)
    net = NestedLinearPolicy(measurement_networks, measurement_network_kwargs, linear_policy_kwargs)
    dummy_input = jp.ones(input_size)
    params = net.init(rng, dummy_input)

    # Print the network summary
    print_module_summary(net, input_size)

    # Forward pass through the network
    my_input = jax.random.normal(rng, input_size)
    my_output = net.apply(params, my_input)
    assert my_output.shape[-1] == output_size[-1]


def test_linear_system_policy():
    """Test creating and evaluating a linear system policy."""
    nz = 4
    ny = 3
    nu = 2

    net = LinearSystemPolicy(nz, ny, nu)
    dummy_input = jp.ones((nz + ny,))  # input is lifted state + observation
    params = net.init(jax.random.PRNGKey(0), dummy_input)

    assert params["params"]["L"].shape == (nz,)
    assert params["params"]["B"].shape == (nz, ny)
    assert params["params"]["C"].shape == (nu, nz)
    assert params["params"]["D"].shape == (nu, ny)

    # Print the network summary
    print_module_summary(net, dummy_input.shape)

    # Forward pass through the network
    my_input = jax.random.normal(jax.random.PRNGKey(0), dummy_input.shape)
    my_output = net.apply(params, my_input)

    assert my_output.shape == (2 * (nz + nu),)


if __name__ == "__main__":
    # test_series()
    # test_parallel()
    # test_hierarchy()
    # test_nested_linear()
    # test_linear_system_policy()
    test_quadratic()
