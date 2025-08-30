// Based upon the work of:
// @InProceedings{HeLiu2021,
//     author    = {Yanhao He and Steven Liu},
//     booktitle = {2021 9th International Conference on Control, Mechatronics and Automation (ICCMA2021)},
//     title     = {Analytical Inverse Kinematics for {F}ranka {E}mika {P}anda -- a Geometrical Solver for
//     7-{DOF} Manipulators with Unconventional Design}, year      = {2021}, month     = nov, publisher =
//     {{IEEE}}, doi       = {10.1109/ICCMA54375.2021.9646185},
// }
// https://github.com/ffall007/franka_analytical_ik

#pragma once

#include <array>
#include <cmath>
#include <Eigen/Dense>

template <typename FloatT>
struct Franka
{
    using Array = std::array<FloatT, 7>;
    using Array4 = std::array<Array, 4>;
    using Matrix4x4 = Eigen::Matrix<FloatT, 4, 4>;
    using Matrix3x3 = Eigen::Matrix<FloatT, 3, 3>;
    using Vector3 = Eigen::Vector<FloatT, 3>;

    static constexpr Array q_NAN = {NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    static constexpr Array4 q_all_NAN = {q_NAN, q_NAN, q_NAN, q_NAN};

    static constexpr FloatT d1 = 0.3330;
    static constexpr FloatT d3 = 0.3160;
    static constexpr FloatT d5 = 0.3840;
    static constexpr FloatT d7e = 0.2104;
    static constexpr FloatT a4 = 0.0825;
    static constexpr FloatT a7 = 0.0880;

    static constexpr FloatT LL24 = a4 * a4 + d3 * d3;
    static constexpr FloatT LL46 = a4 * a4 + d5 * d5;
    static constexpr FloatT L24 = std::sqrt(LL24);
    static constexpr FloatT L46 = std::sqrt(LL46);

    static constexpr FloatT thetaH46 = std::atan(d5 / a4);
    static constexpr FloatT theta342 = std::atan(d3 / a4);
    static constexpr FloatT theta46H = M_PI / 2 - std::atan(d5 / a4);

    static constexpr Array q_min = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
    static constexpr Array q_max = {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};

    static constexpr FloatT NEAR_ONE = 0.999;

    static auto
    ik(const std::array<FloatT, 16> &O_T_EE_array, FloatT q7, const Array &q_actual_array) noexcept -> Array4
    {
        Array4 q_all = q_all_NAN;

        const Eigen::Map<const Matrix4x4> O_T_EE(O_T_EE_array.data());

        if (q7 <= q_min[6] or q7 >= q_max[6])
        {
            return q_all;
        }

        for (auto i = 0U; i < 4; ++i)
        {
            q_all[i][6] = q7;
        }

        // compute p_6
        const Matrix3x3 R_EE = O_T_EE.template topLeftCorner<3, 3>();
        const Vector3 z_EE = O_T_EE.template block<3, 1>(0, 2);
        const Vector3 p_EE = O_T_EE.template block<3, 1>(0, 3);
        const Vector3 p_7 = p_EE - d7e * z_EE;

        const Vector3 x_EE_6(std::cos(q7 - M_PI_4), -std::sin(q7 - M_PI_4), 0.0);
        const Vector3 x_6 = (R_EE * x_EE_6).normalized();
        const Vector3 p_6 = p_7 - a7 * x_6;

        // compute q4
        const Vector3 p_2(0.0, 0.0, d1);
        const Vector3 V26 = p_6 - p_2;

        const FloatT LL26 = V26[0] * V26[0] + V26[1] * V26[1] + V26[2] * V26[2];
        const FloatT L26 = std::sqrt(LL26);

        if (L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24)
        {
            return q_all_NAN;
        }

        const FloatT theta246 = std::acos((LL24 + LL46 - LL26) / 2.0 / L24 / L46);
        const FloatT q4 = theta246 + thetaH46 + theta342 - 2.0 * M_PI;
        if (q4 <= q_min[3] or q4 >= q_max[3])
        {
            return q_all_NAN;
        }

        for (auto i = 0U; i < 4; ++i)
        {
            q_all[i][3] = q4;
        }

        // compute q6
        const FloatT theta462 = std::acos((LL26 + LL46 - LL24) / 2.0 / L26 / L46);
        const FloatT theta26H = theta46H + theta462;
        const FloatT D26 = -L26 * std::cos(theta26H);

        const Vector3 Z_6 = z_EE.cross(x_6);
        const Vector3 Y_6 = Z_6.cross(x_6);
        Matrix3x3 R_6;
        R_6 << x_6, Y_6.normalized(), Z_6.normalized();

        const Vector3 V_6_62 = R_6.transpose() * (-V26);

        const FloatT Phi6 = std::atan2(V_6_62[1], V_6_62[0]);
        const FloatT Theta6 = std::asin(D26 / std::sqrt(V_6_62[0] * V_6_62[0] + V_6_62[1] * V_6_62[1]));

        std::array<FloatT, 2> q6{M_PI - Theta6 - Phi6, Theta6 - Phi6};

        for (auto i = 0U; i < 2; ++i)
        {
            if (q6[i] <= q_min[5])
            {
                q6[i] += 2.0 * M_PI;
            }
            else if (q6[i] >= q_max[5])
            {
                q6[i] -= 2.0 * M_PI;
            }

            if (q6[i] <= q_min[5] or q6[i] >= q_max[5])
            {
                q_all[2 * i] = q_NAN;
                q_all[2 * i + 1] = q_NAN;
            }
            else
            {
                q_all[2 * i][5] = q6[i];
                q_all[2 * i + 1][5] = q6[i];
            }
        }

        if (std::isnan(q_all[0][5]) and std::isnan(q_all[2][5]))
        {
            return q_all_NAN;
        }

        // compute q1 & q2
        const FloatT thetaP26 = 3.0 * M_PI_2 - theta462 - theta246 - theta342;
        const FloatT thetaP = M_PI - thetaP26 - theta26H;
        const FloatT LP6 = L26 * sin(thetaP26) / std::sin(thetaP);

        std::array<Vector3, 4> z_5_all;
        std::array<Vector3, 4> V2P_all;

        for (auto i = 0U; i < 2; ++i)
        {
            const Vector3 z_6_5(std::sin(q6[i]), std::cos(q6[i]), 0.0);
            const Vector3 z_5 = R_6 * z_6_5;
            const Vector3 V2P = p_6 - LP6 * z_5 - p_2;

            z_5_all[2 * i] = z_5;
            z_5_all[2 * i + 1] = z_5;
            V2P_all[2 * i] = V2P;
            V2P_all[2 * i + 1] = V2P;

            const FloatT L2P = V2P.norm();

            if (std::fabs(V2P[2] / L2P) > NEAR_ONE)
            {
                q_all[2 * i][0] = q_actual_array[0];
                q_all[2 * i][1] = 0.0;
                q_all[2 * i + 1][0] = q_actual_array[0];
                q_all[2 * i + 1][1] = 0.0;
            }
            else
            {
                q_all[2 * i][0] = std::atan2(V2P[1], V2P[0]);
                q_all[2 * i][1] = std::acos(V2P[2] / L2P);
                q_all[2 * i + 1][0] = q_all[2 * i][0] + (q_all[2 * i][0] < 0 ? M_PI : -M_PI);
                q_all[2 * i + 1][1] = -q_all[2 * i][1];
            }
        }

        for (auto i = 0U; i < 4; ++i)
        {
            if (q_all[i][0] <= q_min[0] or q_all[i][0] >= q_max[0] or q_all[i][1] <= q_min[1] or
                q_all[i][1] >= q_max[1])
            {
                q_all[i] = q_NAN;
                continue;
            }

            // compute q3
            const Vector3 z_3 = V2P_all[i].normalized();
            const Vector3 Y_3 = -V26.cross(V2P_all[i]);
            const Vector3 y_3 = Y_3.normalized();
            const Vector3 x_3 = y_3.cross(z_3);
            const FloatT c1 = std::cos(q_all[i][0]);
            const FloatT s1 = std::sin(q_all[i][0]);
            Matrix3x3 R_1;
            R_1 << c1, -s1, 0.0, s1, c1, 0.0, 0.0, 0.0, 1.0;

            const FloatT c2 = std::cos(q_all[i][1]);
            const FloatT s2 = std::sin(q_all[i][1]);
            Matrix3x3 R_1_2;
            R_1_2 << c2, -s2, 0.0, 0.0, 0.0, 1.0, -s2, -c2, 0.0;

            const Matrix3x3 R_2 = R_1 * R_1_2;
            const Vector3 x_2_3 = R_2.transpose() * x_3;
            q_all[i][2] = std::atan2(x_2_3[2], x_2_3[0]);

            if (q_all[i][2] <= q_min[2] or q_all[i][2] >= q_max[2])
            {
                q_all[i] = q_NAN;
                continue;
            }

            // compute q5
            const Vector3 VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5_all[i];
            const FloatT c6 = std::cos(q_all[i][5]);
            const FloatT s6 = std::sin(q_all[i][5]);
            Matrix3x3 R_5_6;
            R_5_6 << c6, -s6, 0.0, 0.0, 0.0, -1.0, s6, c6, 0.0;
            const Matrix3x3 R_5 = R_6 * R_5_6.transpose();
            const Vector3 V_5_H4 = R_5.transpose() * VH4;

            q_all[i][4] = -std::atan2(V_5_H4[1], V_5_H4[0]);
            if (q_all[i][4] <= q_min[4] or q_all[i][4] >= q_max[4])
            {
                q_all[i] = q_NAN;
                continue;
            }
        }

        return q_all;
    }

    inline static auto fk(const Array &q_actual_array) noexcept -> std::array<Matrix4x4, 7>
    {
        std::array<FloatT, 6> c, s;
        for (auto i = 0U; i < 6; ++i)
        {
            c[i] = std::cos(q_actual_array[i]);
            s[i] = std::sin(q_actual_array[i]);
        }

        std::array<Matrix4x4, 7> As_a;

        As_a[0] << c[0], -s[0], 0.0, 0.0,  //
            s[0], c[0], 0.0, 0.0,          //
            0.0, 0.0, 1.0, d1,             //
            0.0, 0.0, 0.0, 1.0;
        As_a[1] << c[1], -s[1], 0.0, 0.0,  //
            0.0, 0.0, 1.0, 0.0,            //
            -s[1], -c[1], 0.0, 0.0,        //
            0.0, 0.0, 0.0, 1.0;
        As_a[2] << c[2], -s[2], 0.0, 0.0,  //
            0.0, 0.0, -1.0, -d3,           //
            s[2], c[2], 0.0, 0.0,          //
            0.0, 0.0, 0.0, 1.0;
        As_a[3] << c[3], -s[3], 0.0, a4,  //
            0.0, 0.0, -1.0, 0.0,          //
            s[3], c[3], 0.0, 0.0,         //
            0.0, 0.0, 0.0, 1.0;
        As_a[4] << 1.0, 0.0, 0.0, -a4,  //
            0.0, 1.0, 0.0, 0.0,         //
            0.0, 0.0, 1.0, 0.0,         //
            0.0, 0.0, 0.0, 1.0;
        As_a[5] << c[4], -s[4], 0.0, 0.0,  //
            0.0, 0.0, 1.0, d5,             //
            -s[4], -c[4], 0.0, 0.0,        //
            0.0, 0.0, 0.0, 1.0;
        As_a[6] << c[5], -s[5], 0.0, 0.0,  //
            0.0, 0.0, -1.0, 0.0,           //
            s[5], c[5], 0.0, 0.0,          //
            0.0, 0.0, 0.0, 1.0;

        // Compute cumulative transformations
        std::array<Matrix4x4, 7> Ts_a;

        Ts_a[0] = As_a[0];
        for (auto j = 1U; j < 7; ++j)
        {
            Ts_a[j] = Ts_a[j - 1] * As_a[j];
        }

        return Ts_a;
    }

    static auto
    cc_ik(const std::array<FloatT, 16> &O_T_EE_array, FloatT q7, const Array &q_actual_array) noexcept
        -> Array
    {
        std::array<FloatT, 7> q;
        const Eigen::Map<const Matrix4x4> O_T_EE(O_T_EE_array.data());

        // return NAN if input q7 is out of range
        if (q7 <= q_min[6] or q7 >= q_max[6])
        {
            return q_NAN;
        }

        q[6] = q7;

        auto Ts_a = fk(q_actual_array);

        // identify q6 case
        const Vector3 V62_a = Ts_a[1].template block<3, 1>(0, 3) - Ts_a[6].template block<3, 1>(0, 3);
        const Vector3 V6H_a = Ts_a[4].template block<3, 1>(0, 3) - Ts_a[6].template block<3, 1>(0, 3);
        const Vector3 Z6_a = Ts_a[6].template block<3, 1>(0, 2);
        bool is_case6_0 = ((V6H_a.cross(V62_a)).transpose() * Z6_a <= 0);

        // identify q1 case
        bool is_case1_1 = (q_actual_array[1] < 0);

        // IK: compute p_6
        const Matrix3x3 R_EE = O_T_EE.template topLeftCorner<3, 3>();
        const Vector3 z_EE = O_T_EE.template block<3, 1>(0, 2);
        const Vector3 p_EE = O_T_EE.template block<3, 1>(0, 3);
        const Vector3 p_7 = p_EE - d7e * z_EE;

        const Vector3 x_EE_6(std::cos(q7 - M_PI_4), -std::sin(q7 - M_PI_4), 0.0);
        const Vector3 x_6 = (R_EE * x_EE_6).normalized();
        const Vector3 p_6 = p_7 - a7 * x_6;

        // IK: compute q4
        const Vector3 p_2(0.0, 0.0, d1);
        const Vector3 V26 = p_6 - p_2;

        const FloatT LL26 = V26[0] * V26[0] + V26[1] * V26[1] + V26[2] * V26[2];
        const FloatT L26 = std::sqrt(LL26);

        if (L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24)
        {
            return q_NAN;
        }

        const FloatT theta246 = std::acos((LL24 + LL46 - LL26) / 2.0 / L24 / L46);
        q[3] = theta246 + thetaH46 + theta342 - 2.0 * M_PI;
        if (q[3] <= q_min[3] or q[3] >= q_max[3])
        {
            return q_NAN;
        }

        // IK: compute q6
        const FloatT theta462 = std::acos((LL26 + LL46 - LL24) / 2.0 / L26 / L46);
        const FloatT theta26H = theta46H + theta462;
        const FloatT D26 = -L26 * std::cos(theta26H);

        const Vector3 Z_6 = z_EE.cross(x_6);
        const Vector3 Y_6 = Z_6.cross(x_6);
        Matrix3x3 R_6;
        R_6.col(0) = x_6;
        R_6.col(1) = Y_6.normalized();
        R_6.col(2) = Z_6.normalized();
        const Vector3 V_6_62 = R_6.transpose() * (-V26);

        const FloatT Phi6 = std::atan2(V_6_62[1], V_6_62[0]);
        const FloatT Theta6 = std::asin(D26 / std::sqrt(V_6_62[0] * V_6_62[0] + V_6_62[1] * V_6_62[1]));

        if (is_case6_0)
        {
            q[5] = M_PI - Theta6 - Phi6;
        }
        else
        {
            q[5] = Theta6 - Phi6;
        }

        if (q[5] <= q_min[5])
        {
            q[5] += 2.0 * M_PI;
        }
        else if (q[5] >= q_max[5])
        {
            q[5] -= 2.0 * M_PI;
        }

        if (q[5] <= q_min[5] or q[5] >= q_max[5])
        {
            return q_NAN;
        }

        // IK: compute q1 & q2
        const FloatT thetaP26 = 3.0 * M_PI_2 - theta462 - theta246 - theta342;
        const FloatT thetaP = M_PI - thetaP26 - theta26H;
        const FloatT LP6 = L26 * sin(thetaP26) / std::sin(thetaP);

        const Vector3 z_6_5(std::sin(q[5]), std::cos(q[5]), 0.0);
        const Vector3 z_5 = R_6 * z_6_5;
        const Vector3 V2P = p_6 - LP6 * z_5 - p_2;

        const FloatT L2P = V2P.norm();

        if (std::fabs(V2P[2] / L2P) > NEAR_ONE)
        {
            q[0] = q_actual_array[0];
            q[1] = 0.0;
        }
        else
        {
            q[0] = std::atan2(V2P[1], V2P[0]);
            q[1] = std::acos(V2P[2] / L2P);
            if (is_case1_1)
            {
                q[0] += (q[0] < 0.0) ? M_PI : -M_PI;
                q[1] = -q[1];
            }
        }

        if (q[0] <= q_min[0] or q[0] >= q_max[0] or q[1] <= q_min[1] or q[1] >= q_max[1])
        {
            return q_NAN;
        }

        // IK: compute q3
        const Vector3 z_3 = V2P.normalized();
        const Vector3 Y_3 = -V26.cross(V2P);
        const Vector3 y_3 = Y_3.normalized();
        const Vector3 x_3 = y_3.cross(z_3);
        const FloatT c1 = std::cos(q[0]);
        const FloatT s1 = std::sin(q[0]);
        Matrix3x3 R_1;
        R_1 << c1, -s1, 0.0, s1, c1, 0.0, 0.0, 0.0, 1.0;

        const FloatT c2 = std::cos(q[1]);
        const FloatT s2 = std::sin(q[1]);
        Matrix3x3 R_1_2;
        R_1_2 << c2, -s2, 0.0, 0.0, 0.0, 1.0, -s2, -c2, 0.0;

        const Matrix3x3 R_2 = R_1 * R_1_2;
        const Vector3 x_2_3 = R_2.transpose() * x_3;
        q[2] = std::atan2(x_2_3[2], x_2_3[0]);

        if (q[2] <= q_min[2] or q[2] >= q_max[2])
        {
            return q_NAN;
        }

        // IK: compute q5
        const Vector3 VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5;
        const FloatT c6 = std::cos(q[5]);
        const FloatT s6 = std::sin(q[5]);
        Matrix3x3 R_5_6;
        R_5_6 << c6, -s6, 0.0, 0.0, 0.0, -1.0, s6, c6, 0.0;
        const Matrix3x3 R_5 = R_6 * R_5_6.transpose();
        const Vector3 V_5_H4 = R_5.transpose() * VH4;

        q[4] = -std::atan2(V_5_H4[1], V_5_H4[0]);
        if (q[4] <= q_min[4] or q[4] >= q_max[4])
        {
            return q_NAN;
        }

        return q;
    }
};
