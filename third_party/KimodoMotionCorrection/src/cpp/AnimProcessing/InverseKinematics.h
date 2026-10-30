/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Math/Transform.h"

#include <vector>

using Pose = std::vector<Math::Transform>;

namespace IK {

    void TwoBoneIk(
        Pose& pose,
        const Math::Transform& rootTransform,
        uint32_t jointIdx,
        float weight,
        const Math::Vector& target,
        const std::vector<int>& joint_parents_vec,
        const Math::Vector& hintOffset = Math::Vector::Zero
    );

    void OneBoneIk(
        Pose& pose,
        const Math::Transform& rootTransform,
        uint32_t jointIdx,
        float weight,
        const Math::Vector& target,
        const std::vector<int>& joint_parents_vec
    );

}
