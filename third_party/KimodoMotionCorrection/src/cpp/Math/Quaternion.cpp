/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Quaternion.h"
#include "Matrix.h"

namespace Math
{
    Quaternion const Quaternion::Identity(0, 0, 0, 1);

    // Rotation order is XYZ
    EulerAngles Quaternion::ToEulerAngles() const
    {
        return Matrix(*this).ToEulerAngles();
    }

    Quaternion Quaternion::LookRotation(const Vector& forward, const Vector& up)
    {
        const Vector t = Vector::Cross3(up, forward).Normalize3();
        return Matrix(t, Vector::Cross3(forward, t), forward).GetRotation();
    }
}
