# FRANTIK: nanobind wrapper for analytic Franka IK

A Python wrapper for an analytical Inverse Kinematics (IK) solver for Franka written with [nanobind](https://nanobind.readthedocs.io/en/latest/).
The solver is modified from [franka_analytic_ik](https://github.com/ffall007/franka_analytical_ik/tree/main), which has the following citation:
```bibtex
@InProceedings{HeLiu2021,
  author    = {Yanhao He and Steven Liu},
  booktitle = {2021 9th International Conference on Control, Mechatronics and Automation (ICCMA2021)},
  title     = {Analytical Inverse Kinematics for {F}ranka {E}mika {P}anda -- a Geometrical Solver for 7-{DOF} Manipulators with Unconventional Design},
  year      = {2021},
  month     = nov,
  publisher = {{IEEE}},
  doi       = {10.1109/ICCMA54375.2021.9646185},
}
```

## Installation

Simply pip install:
```bash
pip install frantik
```

You can also clone the repository and pip install:
```bash
git clone git@github.com:CoMMALab/frantik.git
cd frantik
pip install .
```

You will need [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) installed.
To install on Ubuntu 22.04, `sudo apt install libeigen3-dev`.

## Usage Notes

This module provides two functions:
- `ik(tf, q7, qc)` which takes as input the target frame (a 4x4 TF matrix), the desired value of joint 7 in radians, and the current configuration of the robot. It returns all four solutions (if they exist) to the IK problem.
- `cc_ik(tf, q7, qc)` which takes in the same input as above, but only returns the solution closest to the current configuration.

Note that the functions take numpy arrays as input.

> [!WARNING]
> Note that both functions assume that Franka Hand is installed and the solution is regarding to the end effector frame. If the Cartesian pose of the flange frame is to be used as input, according to Franka documentation you can change the const variable d7e to 0.107 and remove the 45 degree offset in q7.

## TODO
- [ ] Line search for q7 values
