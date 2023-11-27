import jax
import jax.numpy as jp

from ambersim.learning.architectures import MLP, HierarchyComposition, ParallelComposition, SeriesComposition


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


if __name__ == "__main__":
    test_series()
    test_parallel()
    test_hierarchy()
