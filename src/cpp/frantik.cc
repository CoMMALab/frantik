#include "ik.hh"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core_ext, pymodule)
{
    pymodule.def("ik", ik_wrapper, "target_transform"_a, "q7"_a, "q_actual"_a, "Position IK for Franka EE.");

    pymodule.def(
        "cc_ik",
        Franka<double>::cc_ik,
        "target_transform"_a,
        "q7"_a,
        "q_actual"_a,
        "Case-consistent position IK for Franka EE (i.e., avoids elbow flips).");
}
