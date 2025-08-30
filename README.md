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

Simply clone the repository and pip install:
```bash
git clone git@github.com:CoMMALab/frantik.git
cd frantik
pip install .
```

You will need [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) installed.
To install on Ubuntu 22.04, `sudo apt install libeigen3-dev`.

## TODO
- [ ] Better interface with numpy
- [ ] Line search for q7 values
