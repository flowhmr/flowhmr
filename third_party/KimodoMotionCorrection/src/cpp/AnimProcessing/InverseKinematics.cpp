/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "InverseKinematics.h"
#include "Math/Scalar.h"
#include <iostream>


using namespace IK;

namespace
{

float getAngleWithTwoSideVectors(const Math::Vector& vecLeft, const Math::Vector& vecRight)
{
    auto lNorm = vecLeft.GetNormalized3();
    auto rNorm = vecRight.GetNormalized3();

    float cosine = lNorm.GetDot3(rNorm);
    float sine = lNorm.Cross3(rNorm).GetLength3();

    return atan2f(sine, cosine);  // in radian
}

float getAngleWithCosineRule (const float lSideLeft, const float lSideRight, const float lSideAcross)
{
    float val =
        (lSideRight * lSideRight + lSideLeft * lSideLeft - lSideAcross * lSideAcross) /
            (2.0f * lSideLeft * lSideRight);
    val = Math::Clamp(val, -1.0f, 1.0f);  // numerical stability. also avoid impossible trangulars
    return acosf(val);  // in radian
}

}


void IK::TwoBoneIk(
    Pose& pose,
    const Math::Transform& rootTransform,
    uint32_t cIdx,
    float weight,
    const Math::Vector& target,
    const std::vector<int>& joint_parents_vec,
    const Math::Vector& hintOffset
)
{
    weight = Math::Clamp(weight, 0.0f, 1.0f);
    if (!(weight > 0.0f))
        return;

    // Two bone IK: joints are represented as "a", "b", "c" in the below comments:
    //  1. stage 1, bend joint a and joint b, so that |ac| = |at|, while vec_ac maintain the same direction
    //  2. stage 2, rotate start joint a so that c and t are in the same place

    //  a                   a                   a             |
    //  |\                  |\                  |\            |
    //  | \                 |  \                | \           |
    //  |  \  (stage 1 ->)  |   \  (stage 2 ->) |  \          |
    //  |   b               |    b              |   b         |
    //  |    \              |    |              |  /          |
    //  |     \             |     |             | /           |
    //  t      c            t      c            t(c)          |
    //  (a is the root joint, b is the middle joint and c is the end joint)
    //

    int32_t bIdx = joint_parents_vec[cIdx];
    if (bIdx < 0)
    {
        return;
    }
    int32_t aIdx = joint_parents_vec[bIdx];
    if (aIdx < 0)
    {
        return;
    }

    // Find the parent world transform of joint a:
    Math::Transform aParentWorldTransform = Math::Transform::Identity;
    int32_t idx = joint_parents_vec[aIdx];
    while (idx >= 0)
    {
        aParentWorldTransform = aParentWorldTransform * pose[idx];
        idx = joint_parents_vec[idx];
    }
    aParentWorldTransform = aParentWorldTransform * rootTransform;

    // Compute world space transforms of a, b and c:
    Math::Transform aWorld = pose[aIdx] * aParentWorldTransform;
    Math::Transform bWorld = pose[bIdx] * aWorld;
    Math::Transform cWorld = pose[cIdx] * bWorld;

    auto a = aWorld.GetTranslation();
    auto b = bWorld.GetTranslation();
    auto c = cWorld.GetTranslation();
    auto t = Math::Vector::Lerp(c, target, weight);

    // step 1 (stage 1): extend / contract the joint chain to match the distance
    float eps = 0.0001f;  // numerical stability
    float l_ab = (b - a).Length3().GetX();
    float l_bc = (c - b).Length3().GetX();
    float l_at = (a - t).Length3().GetX();
    l_at = Math::Clamp(l_at, eps, (l_ab + l_bc) * 0.999f); // when not reachable, replace with maximum reachable length

    // get the current angles
    float theta_bac_current = getAngleWithTwoSideVectors(a - b, a - c);
    float theta_abc_current = getAngleWithTwoSideVectors(b - a, b - c);
    // get the desired angles
    if (l_ab < eps || l_bc < eps || l_at < eps)
    {
        return;  // the length is too small. rejecting potentially numerically unstable requests.
    }

    // RGB MoCap supplies zero hints to retain the observed SMPL-X bend plane.
    // That plane is undefined on a straight leg: tiny signed roundoff must not
    // turn a newly flexed support knee back and forth. For this singular case
    // only, solve the two segment positions with anatomical local +Z as pole.
    // Non-singular observed planes continue through the original solver below.
    const auto observedAxis = Math::Vector::Cross3(c - a, b - a);
    if (hintOffset.GetLength3() == 0.f && observedAxis.GetLength3() < 0.02f * l_ab * l_bc)
    {
        auto direction = t - a;
        if (direction.GetLength3() < eps) direction = b - a;
        direction = direction.GetNormalized3();
        float distance = Math::Clamp(l_at, Math::Abs(l_ab - l_bc) + eps, (l_ab + l_bc) * 0.999f);
        auto bend = bWorld.GetRotation().RotateVector(Math::Vector(0, 0, 1));
        bend = bend - direction * bend.GetDot3(direction);
        if (bend.GetLength3() < eps)
        {
            bend = bWorld.GetRotation().RotateVector(Math::Vector(1, 0, 0));
            bend = bend - direction * bend.GetDot3(direction);
        }
        bend = bend.GetNormalized3();
        float along = (l_ab * l_ab + distance * distance - l_bc * l_bc) / (2.f * distance);
        float height = std::sqrt(Math::Max(0.f, l_ab * l_ab - along * along));
        auto desiredB = a + direction * along + bend * height;
        auto desiredC = a + direction * distance;
        auto fromA = aWorld.GetRotation().RotateVectorInverse(b - a);
        auto toA = aWorld.GetRotation().RotateVectorInverse(desiredB - a);
        pose[aIdx] = Math::Transform(Math::Quaternion::FromRotationBetweenVectors(fromA, toA), Math::Vector::Zero) * pose[aIdx];
        aWorld = pose[aIdx] * aParentWorldTransform;
        bWorld = pose[bIdx] * aWorld;
        cWorld = pose[cIdx] * bWorld;
        auto fromB = bWorld.GetRotation().RotateVectorInverse(cWorld.GetTranslation() - bWorld.GetTranslation());
        auto toB = bWorld.GetRotation().RotateVectorInverse(desiredC - bWorld.GetTranslation());
        pose[bIdx] = Math::Transform(Math::Quaternion::FromRotationBetweenVectors(fromB, toB), Math::Vector::Zero) * pose[bIdx];
        return;
    }
    float theta_bac_desired = getAngleWithCosineRule(l_ab, l_at, l_bc);
    float theta_abc_desired = getAngleWithCosineRule(l_ab, l_bc, l_at);

    // in joint[0]'s parent's space
    Math::Vector rotationAxis = Math::Vector::Cross3(c - a, bWorld.TransformPoint(hintOffset) - a);
    float l = rotationAxis.GetLength3();
    if (l == 0)
    {
        rotationAxis = Math::Vector(0,0,1);
    }
    else
    {
        rotationAxis /= l;
    }

    // get the rotation with axis in the local space of joint a and joint b
    Math::Vector rotationAxisLocalInBSpace = bWorld.GetRotation().RotateVectorInverse(rotationAxis);
    Math::Transform rotateInB(
        Math::Quaternion(rotationAxisLocalInBSpace,
            (theta_abc_desired - theta_abc_current)), Math::Vector::Zero);

    pose[bIdx] = rotateInB * pose[bIdx];

    Math::Vector rotationAxisLocalInASpace = aWorld.GetRotation().RotateVectorInverse(rotationAxis);
    Math::Transform rotateInA(
        Math::Quaternion(rotationAxisLocalInASpace,
            (theta_bac_desired - theta_bac_current)), Math::Vector::Zero);

    pose[aIdx] = rotateInA * pose[aIdx];

    // recompute a's world space transform as we're going to need it:
    aWorld = pose[aIdx] * aParentWorldTransform;

    // step 2 (stage 2): rotate joint a so that the target and the end joint c matches
    auto acLocal = aWorld.GetRotation().RotateVectorInverse(
        c - a);
    auto atLocal = aWorld.GetRotation().RotateVectorInverse(
        target - a);
    Math::Transform rotateStageTwo(
        Math::Quaternion::FromRotationBetweenVectors(acLocal, atLocal), Math::Vector::Zero
    );

    pose[aIdx] = rotateStageTwo * pose[aIdx];

}

void IK::OneBoneIk(
    Pose& pose,
    const Math::Transform& rootTransform,
    uint32_t bIdx,
    float weight,
    const Math::Vector& target,
    const std::vector<int>& joint_parents_vec
)
{
    weight = Math::Clamp(weight, 0.0f, 1.0f);
    if (!(weight > 0.0f))
        return;

    int32_t aIdx = joint_parents_vec[bIdx];
    if (aIdx < 0)
    {
        return;
    }

    // Find the parent world transform of joint a:
    Math::Transform aParentWorldTransform = Math::Transform::Identity;
    int32_t idx = joint_parents_vec[aIdx];
    while (idx >= 0)
    {
        aParentWorldTransform = aParentWorldTransform * pose[idx];
        idx = joint_parents_vec[idx];
    }
    aParentWorldTransform = aParentWorldTransform * rootTransform;

    // Compute world space transforms of a, b and c:
    Math::Transform aWorld = pose[aIdx] * aParentWorldTransform;
    Math::Transform bWorld = pose[bIdx] * aWorld;

    auto abLocal = aWorld.GetRotation().RotateVectorInverse(
        bWorld.GetTranslation() - aWorld.GetTranslation());
    auto atLocal = aWorld.GetRotation().RotateVectorInverse(
        target - aWorld.GetTranslation());

    auto deltaRLocal = Math::Quaternion::NLerp(Math::Quaternion::Identity, Math::Quaternion::FromRotationBetweenVectors(abLocal, atLocal), weight);
    pose[aIdx].SetRotation(deltaRLocal * pose[aIdx].GetRotation());
}
