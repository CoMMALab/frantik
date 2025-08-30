#include "ik.hh"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename FloatT>
static auto ik_wrapper(
    nb::ndarray<FloatT, nb::numpy, nb::shape<4, 4>, nb::device::cpu> target_transform,
    FloatT q7,
    nb::ndarray<FloatT, nb::numpy, nb::shape<7>, nb::device::cpu> q_actual)
{
    const Eigen::Map<const Eigen::Matrix<FloatT, 4, 4, Eigen::RowMajor>> O_T_EE(target_transform.data());

    auto result = Franka<FloatT>::ik(O_T_EE, q7, q_actual.data());

    auto *arr = new FloatT[4 * 7];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 7; ++j)
        {
            arr[i * 7 + j] = result[i][j];
        }
    }

    nb::capsule arr_owner(arr, [](void *a) noexcept { delete[] reinterpret_cast<FloatT *>(a); });
    return nb::ndarray<FloatT, nb::numpy, nb::shape<4, 7>, nb::device::cpu>(arr, {4, 7}, arr_owner);
}

template <typename FloatT>
static auto cc_ik_wrapper(
    nb::ndarray<FloatT, nb::numpy, nb::shape<4, 4>, nb::device::cpu> target_transform,
    FloatT q7,
    nb::ndarray<FloatT, nb::numpy, nb::shape<7>, nb::device::cpu> q_actual)
{
    const Eigen::Map<const Eigen::Matrix<FloatT, 4, 4, Eigen::RowMajor>> O_T_EE(target_transform.data());

    auto result = Franka<FloatT>::cc_ik(O_T_EE, q7, q_actual.data());

    auto *arr = new FloatT[7];
    for (int i = 0; i < 7; ++i)
    {
        arr[i] = result[i];
    }

    nb::capsule arr_owner(arr, [](void *a) noexcept { delete[] reinterpret_cast<FloatT *>(a); });
    return nb::ndarray<FloatT, nb::numpy, nb::shape<7>, nb::device::cpu>(arr, {7}, arr_owner);
}

NB_MODULE(_core_ext, pymodule)
{
    pymodule.def(
        "ik", ik_wrapper<double>, "target_transform"_a, "q7"_a, "q_actual"_a, "Position IK for Franka EE.");

    pymodule.def(
        "cc_ik",
        cc_ik_wrapper<double>,
        "target_transform"_a,
        "q7"_a,
        "q_actual"_a,
        "Case-consistent position IK for Franka EE (i.e., avoids elbow flips).");
}
