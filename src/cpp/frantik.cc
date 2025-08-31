#include "ik.hh"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename FloatT>
struct Helper
{
    using NP_TF = nb::ndarray<FloatT, nb::numpy, nb::shape<4, 4>, nb::device::cpu>;
    using NP_C = nb::ndarray<FloatT, nb::numpy, nb::shape<7>, nb::device::cpu>;

    static auto to_eigen(NP_TF &tf) -> Eigen::Matrix<FloatT, 4, 4, Eigen::RowMajor>
    {
        return Eigen::Map<const Eigen::Matrix<FloatT, 4, 4, Eigen::RowMajor>>(tf.data());
    }

    static auto ik(NP_TF target_transform, FloatT q7, NP_C q_c)
    {
        auto tf = to_eigen(target_transform);
        auto result = Franka<FloatT>::ik(tf, q7, q_c.data());

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

    static auto cc_ik(NP_TF target_transform, FloatT q7, NP_C q_c)
    {
        auto tf = to_eigen(target_transform);
        auto result = Franka<FloatT>::cc_ik(tf, q7, q_c.data());

        auto *arr = new FloatT[7];
        for (int i = 0; i < 7; ++i)
        {
            arr[i] = result[i];
        }

        nb::capsule arr_owner(arr, [](void *a) noexcept { delete[] reinterpret_cast<FloatT *>(a); });
        return nb::ndarray<FloatT, nb::numpy, nb::shape<7>, nb::device::cpu>(arr, {7}, arr_owner);
    }

    static auto ik7(NP_TF target_transform, NP_C q_c)
    {
        auto tf = to_eigen(target_transform);
        auto result = Franka<FloatT>::ik_q7(tf, q_c.data());

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

    static auto cc_ik7(NP_TF target_transform, NP_C q_c)
    {
        auto tf = to_eigen(target_transform);
        auto result = Franka<FloatT>::cc_ik_q7(tf, q_c.data());

        auto *arr = new FloatT[7];
        for (int i = 0; i < 7; ++i)
        {
            arr[i] = result[i];
        }

        nb::capsule arr_owner(arr, [](void *a) noexcept { delete[] reinterpret_cast<FloatT *>(a); });
        return nb::ndarray<FloatT, nb::numpy, nb::shape<7>, nb::device::cpu>(arr, {7}, arr_owner);
    }
};

NB_MODULE(_core_ext, pymodule)
{
    pymodule.def(
        "ik", Helper<double>::ik, "target_transform"_a, "q7"_a, "q_c"_a, "Position IK for Franka EE.");

    pymodule.def(
        "cc_ik",
        Helper<double>::cc_ik,
        "target_transform"_a,
        "q7"_a,
        "q_c"_a,
        "Case-consistent position IK for Franka EE (i.e., avoids elbow flips).");

    pymodule.def(
        "ik7",
        Helper<double>::ik7,
        "target_transform"_a,
        "q_c"_a,
        "Position IK for Franka EE. Uses nearest value of q7.");

    pymodule.def(
        "cc_ik7",
        Helper<double>::cc_ik7,
        "target_transform"_a,
        "q_c"_a,
        "Case-consistent position IK for Franka EE (i.e., avoids elbow flips). Uses nearest value of q7.");
}
