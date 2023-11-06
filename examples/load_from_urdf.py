import mujoco as mj
from mujoco import mjx

from ambersim import ROOT
from ambersim.utils import load_mjx_model_from_file

"""This example demonstrates how to load robots from URDFs into mujoco/mjx."""

# all of the following work:
# (1) a global path (ROOT specifies the repository root globally)
# (2) a local path
# (3) a path specified with respect to the repository root
mjx_model1, mjx_data1 = load_mjx_model_from_file(ROOT + "/models/pendulum/pendulum.urdf")  # (1)
mjx_model3, mjx_data3 = load_mjx_model_from_file("models/pendulum/pendulum.urdf")  # (3)
