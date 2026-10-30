/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Transform.h"

namespace Math
{
    Transform const Transform::Identity = Transform(Quaternion(0, 0, 0, 1), Vector(0, 0, 0, 1), 1.0f);

    void Transform::SanitizeScaleValue()
    {
        if (Math::IsNearEqual(GetScale(), 1.0f, Math::LargeEpsilon))
        {
            SetScale(1.0f);
        }
    }
}
