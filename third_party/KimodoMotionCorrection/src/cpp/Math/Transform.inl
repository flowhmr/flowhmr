/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Transform.h"

namespace Math
{
    inline Transform Transform::FromRotation(const Quaternion& rotation)
    {
        return Transform(rotation);
    }

    inline Transform Transform::FromTranslation(const Vector& translation)
    {
        return Transform(Quaternion::Identity, translation);
    }

    inline Transform Transform::FromScale(float uniformScale)
    {
        return Transform(Quaternion::Identity, Vector::Zero, uniformScale);
    }

    inline Transform Transform::FromTranslationAndScale(const Vector& translation, float uniformScale)
    {
        return Transform(Quaternion::Identity, translation, uniformScale);
    }

    inline Transform Transform::FromRotationBetweenVectors(const Vector sourceVector, const Vector targetVector)
    {
        return Transform(Quaternion::FromRotationBetweenNormalizedVectors(sourceVector, targetVector));
    }

    inline Transform Transform::Lerp(const Transform& from, const Transform& to, float t)
    {
        Quaternion const rotation = Quaternion::NLerp(Quaternion(from.m_rotation), Quaternion(to.m_rotation), t);
        Vector const translationAndScale = Vector::Lerp(from.m_translationScale, to.m_translationScale, t);

        Transform lerped(NoInit);
        lerped.m_rotation = rotation;
        lerped.m_translationScale = translationAndScale;
        return lerped;
    }

    inline Transform Transform::Slerp(const Transform& from, const Transform& to, float t)
    {
        Quaternion const rotation = Quaternion::SLerp(Quaternion(from.m_rotation), Quaternion(to.m_rotation), t);
        Vector const translationAndScale = Vector::Lerp(Vector(from.m_translationScale), Vector(to.m_translationScale), t);

        Transform lerped(NoInit);
        lerped.m_rotation = rotation;
        lerped.m_translationScale = translationAndScale;
        return lerped;
    }

    inline Transform Transform::FastSlerp(const Transform& from, const Transform& to, float t)
    {
        Quaternion const rotation = Quaternion::FastSLerp(Quaternion(from.m_rotation), Quaternion(to.m_rotation), t);
        Vector const translationAndScale = Vector::Lerp(Vector(from.m_translationScale), Vector(to.m_translationScale), t);

        Transform lerped(NoInit);
        lerped.m_rotation = rotation;
        lerped.m_translationScale = translationAndScale;
        return lerped;
    }

    inline Transform Transform::Delta(const Transform& from, const Transform& to)
    {
        ASSERT(from.m_rotation.IsNormalized() && to.m_rotation.IsNormalized());
        ASSERT(!from.m_translationScale.IsW0() && !to.m_translationScale.IsW0());

        Transform result;

        Vector const inverseScale = from.GetInverseScaleVector();
        Vector const deltaScale = to.GetScaleVector() * inverseScale;

        // If we have negative scaling, we need to use matrices to calculate the deltas
        Vector const minScale = Vector::Min(from.m_translationScale.GetSplatW(), to.m_translationScale.GetSplatW());
        if (minScale.IsAnyLessThan(Vector::Zero))
        {
            // Multiply the transforms using matrices to get the correct rotation and then remove the scale;
            Matrix const toMtx = to.ToMatrix();
            Matrix const fromMtx = from.ToMatrix();
            Matrix resultMtx = toMtx * fromMtx.GetInverse();
            resultMtx.RemoveScaleFast();

            // Apply back the signs from the final scale
            Vector const sign = deltaScale.GetSign();
            resultMtx[0] *= sign.GetSplatX();
            resultMtx[1] *= sign.GetSplatY();
            resultMtx[2] *= sign.GetSplatZ();

            result.m_rotation = resultMtx.GetRotation();
            ASSERT(result.m_rotation.IsNormalized());
            result.m_translationScale = Vector::Select(resultMtx.GetTranslation(), deltaScale, Vector::Select0001);
        }
        else
        {
            Quaternion const fromInverseRotation = from.m_rotation.GetInverse();
            result.m_rotation = to.m_rotation * fromInverseRotation;

            Vector const deltaTranslation = to.m_translationScale - from.m_translationScale;
            Vector const translation = fromInverseRotation.RotateVector(deltaTranslation) * inverseScale;
            result.m_translationScale = Vector::Select(translation, deltaScale, Vector::Select0001);
        }

        return result;
    }

    inline Transform Transform::DeltaNoScale(const Transform& from, const Transform& to)
    {
        Quaternion const inverseFromRotation = from.m_rotation.GetInverse();
        Vector const deltaTranslation = to.GetTranslation() - from.GetTranslation();

        Transform delta;
        delta.m_rotation = to.m_rotation * inverseFromRotation;
        delta.m_translationScale = inverseFromRotation.RotateVector(deltaTranslation).GetWithW1();
        return delta;
    }

    inline void Transform::DirectlySetRotation(Transform& transform, Quaternion&& rotation)
    {
        transform.m_rotation = rotation;
    }

    inline void Transform::DirectlySetRotation(Transform& transform, const Quaternion& rotation)
    {
        transform.m_rotation = rotation;
    }

    inline void Transform::DirectlySetTranslationScale(Transform& transform, Vector&& translationScale)
    {
        transform.m_translationScale = translationScale;
    }

    inline void Transform::DirectlySetTranslationScale(Transform& transform, const Vector& translationScale)
    {
        transform.m_translationScale = translationScale;
    }

    inline Transform::Transform(NoInit_t)
        : m_rotation(NoInit)
        , m_translationScale(NoInit)
    {
    }

    inline Transform::Transform(const Matrix& m)
    {
        Vector mTranslation, mScale;
        m.Decompose(m_rotation, mTranslation, mScale);
        ASSERT(Math::IsNearEqual(mScale.GetX(), mScale.GetY()) && Math::IsNearEqual(mScale.GetY(),mScale.GetZ()));
        m_translationScale = Vector::Select(mTranslation, mScale, Vector::Select0001);
    }

    inline Transform::Transform(const Quaternion& rotation, const Vector& translation, float scale)
        : m_rotation(rotation)
        , m_translationScale(Vector::Select(translation, Vector(scale), Vector::Select0001))
    {
    }

    inline Transform::Transform(const AxisAngle& rotation)
        : m_rotation(rotation)
        , m_translationScale(Vector::UnitW)
    {
    }

    inline Matrix Transform::ToMatrix() const
    {
        return Matrix(m_rotation, m_translationScale.GetWithW1(), m_translationScale.GetSplatW());
    }

    inline Matrix Transform::ToMatrixNoScale() const
    {
        return Matrix(m_rotation, m_translationScale.GetWithW1(), Vector::One);
    }

    inline EulerAngles Transform::ToEulerAngles() const
    {
        return m_rotation.ToEulerAngles();
    }

    inline Vector Transform::GetAxisX() const
    {
        return m_rotation.RotateVector(Vector::UnitX);
    }

    inline Vector Transform::GetAxisY() const
    {
        return m_rotation.RotateVector(Vector::UnitY);
    }

    inline Vector Transform::GetAxisZ() const
    {
        return m_rotation.RotateVector(Vector::UnitZ);
    }

    inline Vector Transform::GetRightVector() const
    {
        return m_rotation.RotateVector(Vector::WorldRight);
    }

    inline Vector Transform::GetForwardVector() const
    {
        return m_rotation.RotateVector(Vector::WorldForward);
    }

    inline Vector Transform::GetUpVector() const
    {
        return m_rotation.RotateVector(Vector::WorldUp);
    }

    inline bool Transform::IsIdentity() const
    {
        return m_rotation.IsIdentity() && m_translationScale.IsEqual4(Vector::UnitW);
    }

    inline bool Transform::IsRigidTransform() const
    {
        return GetScale() == 1.0f;
    }

    inline void Transform::MakeRigidTransform()
    {
        SetScale(1.0f);
    }

    inline Transform& Transform::Inverse()
    {
        ASSERT(!m_translationScale.IsW0());

        Quaternion const inverseRotation = m_rotation.GetInverse();
        m_rotation = inverseRotation;

        Vector const inverseScale = GetInverseScaleVector();
        Vector const inverselyScaledTranslation = inverseScale * m_translationScale.GetWithW0();
        Vector const inverselyRotatedTranslation = inverseRotation.RotateVector(inverselyScaledTranslation);
        Vector const inverseTranslation = inverselyRotatedTranslation.GetNegated().SetW0();

        m_translationScale = Vector::Select(inverseTranslation, inverseScale, Vector::Select0001);

        return *this;
    }

    inline Transform Transform::GetInverse() const
    {
        Transform inverse = *this;
        return inverse.Inverse();
    }

    inline Transform Transform::GetDeltaToOther(const Transform& targetTransform) const
    {
        return Transform::Delta(*this, targetTransform);
    }

    inline Transform Transform::GetDeltaFromOther(const Transform& startTransform) const
    {
        return Transform::Delta(startTransform, *this);
    }

    inline const Quaternion& Transform::GetRotation() const
    {
        return m_rotation;
    }

    inline void Transform::SetRotation(const Quaternion& rotation)
    {
        ASSERT(rotation.IsNormalized());
        m_rotation = rotation;
    }

    inline void Transform::AddRotation(const Quaternion& delta)
    {
        ASSERT(delta.IsNormalized());
        m_rotation = delta * m_rotation;
    }

    inline const Vector& Transform::GetTranslation() const
    {
        return m_translationScale;
    }

    inline const Vector& Transform::GetTranslationAndScale() const
    {
        return m_translationScale;
    }

    inline void Transform::SetTranslation(const Vector& newTranslation)
    {
        m_translationScale = Vector::Select(newTranslation, m_translationScale, Vector::Select0001);
    }

    inline void Transform::SetTranslationAndScale(const Vector& newTranslationScale)
    {
        ASSERT(newTranslationScale.GetW() != 0.0f);
        m_translationScale = newTranslationScale;
    }

    inline void Transform::AddTranslation(const Vector& translationDelta)
    {
        m_translationScale += translationDelta.GetWithW0();
    }

    inline Vector Transform::GetTranslationAsVector() const
    {
        return m_translationScale.GetWithW0();
    }

    inline Vector Transform::GetTranslationAsPoint() const
    {
        return m_translationScale.GetWithW1();
    }

    inline float Transform::GetScale() const
    {
        return m_translationScale.GetW();
    }

    inline Vector Transform::GetScaleVector() const
    {
        return m_translationScale.GetSplatW();
    }

    inline Vector Transform::GetInverseScaleVector() const
    {
        return m_translationScale.GetSplatW().GetInverse();
    }

    inline void Transform::SetScale(float uniformScale)
    {
        m_translationScale.SetW(uniformScale);
    }

    inline bool Transform::HasScale() const
    {
        return m_translationScale.GetW() != 1.0f;
    }

    inline bool Transform::HasNegativeScale() const
    {
        return m_translationScale.GetW() < 0.0f;
    }

    inline Vector Transform::TranslateVector(const Vector& vector) const
    {
        return vector + m_translationScale.GetWithW0();
    }

    inline Vector Transform::ScaleVector(const Vector& vector) const
    {
        return vector * GetScaleVector();
    }

    inline Vector Transform::TransformPoint(const Vector& point) const
    {
        ASSERT(!m_translationScale.IsW0());
        Vector transformedPoint = point * m_translationScale.GetSplatW();
        transformedPoint = (m_translationScale + m_rotation.RotateVector(transformedPoint)).GetWithW0();
        return transformedPoint;
    }

    inline Vector Transform::TransformPointNoScale(const Vector& point) const
    {
        Vector transformedPoint = (m_translationScale + m_rotation.RotateVector(point)).GetWithW0();;
        return transformedPoint;
    }

    inline Vector Transform::RotateVector(const Vector& vector) const
    {
        return m_rotation.RotateVector(vector);
    }

    inline Vector Transform::TransformNormal(const Vector& vector) const
    {
        return RotateVector(vector);
    }

    inline Vector Transform::InverseRotateVector(const Vector& vector) const
    {
        return m_rotation.RotateVectorInverse(vector);
    }

    inline Vector Transform::InverseTransformPoint(const Vector& point) const
    {
        ASSERT(!m_translationScale.IsW0());
        Vector const shiftedPoint = point - m_translationScale;
        Vector const unrotatedShiftedPoint = m_rotation.RotateVectorInverse(shiftedPoint);
        Vector const inverseScale = GetInverseScaleVector();
        Vector const result = unrotatedShiftedPoint * inverseScale;
        return result;
    }

    inline Vector Transform::InverseTransformPointNoScale(const Vector& point) const
    {
        Vector const shiftedPoint = point - m_translationScale;
        Vector const unrotatedShiftedPoint = m_rotation.RotateVectorInverse(shiftedPoint);
        return unrotatedShiftedPoint;
    }

    inline Vector Transform::TransformVector(const Vector& vector) const
    {
        ASSERT(!m_translationScale.IsW0());
        Vector transformedVector = vector * GetScaleVector();
        transformedVector = m_rotation.RotateVector(transformedVector);
        return transformedVector;
    }

    inline Vector Transform::TransformVectorNoScale(const Vector& vector) const
    {
        return RotateVector(vector);
    }

    inline Vector Transform::InverseTransformVector(const Vector& vector) const
    {
        ASSERT(!m_translationScale.IsW0());
        Vector const unrotatedVector = m_rotation.RotateVectorInverse(vector);
        Vector const inverseScale = GetInverseScaleVector();
        Vector const result = unrotatedVector * inverseScale;
        return result;
    }

    inline Vector Transform::InverseTransformVectorNoScale(const Vector& vector) const
    {
        return m_rotation.RotateVectorInverse(vector);
    }

    inline Transform Transform::operator*(const Transform& rhs) const
    {
        Transform transform = *this;
        transform *= rhs;
        return transform;
    }

    inline Transform& Transform::operator*=(const Transform& rhs)
    {
        Vector const scale = GetScaleVector();
        Vector const rhsScale = rhs.GetScaleVector();
        Vector const minScale = Vector::Min(scale, rhsScale);
        Vector const finalScale = scale * rhsScale;

        if (minScale.IsAnyLessThan(Vector::Zero))
        {
            // Multiply the transforms using matrices to
            // get the correct rotation and then remove the scale;
            Matrix const lhsMtx = ToMatrix();
            Matrix const rhsMtx = rhs.ToMatrix();
            Matrix resultMtx = lhsMtx * rhsMtx;
            resultMtx.RemoveScaleFast();

            // Apply back the signs from the final scale
            Vector const sign = finalScale.GetSign();
            resultMtx[0] *= sign.GetSplatX();
            resultMtx[1] *= sign.GetSplatY();
            resultMtx[2] *= sign.GetSplatZ();

            m_rotation = resultMtx.GetRotation();
            ASSERT(m_rotation.IsNormalized());
            m_translationScale = Vector::Select(resultMtx.GetTranslation(), finalScale, Vector::Select0001);
        }
        else
        {
            // Normal case
            m_rotation = m_rotation * rhs.m_rotation;
            m_rotation.Normalize();
            Vector const translation = rhs.m_rotation.RotateVector(m_translationScale * rhsScale) + rhs.m_translationScale;
            m_translationScale = Vector::Select(translation, finalScale, Vector::Select0001);
        }

        return *this;
    }

    inline bool Transform::IsNearEqual(const Transform& rhs, const Radians angleThreshold, float translationScaleThreshold) const
    {
        if (!m_rotation.IsNearEqual(rhs.m_rotation, angleThreshold))
        {
            return false;
        }

        if (!m_translationScale.IsNearEqual4(rhs.m_translationScale, translationScaleThreshold))
        {
            return false;
        }

        return true;
    }

    inline bool Transform::operator==(const Transform& rhs) const
    {
        if (m_translationScale != rhs.m_translationScale)
        {
            return false;
        }

        if (m_rotation != rhs.m_rotation)
        {
            return false;
        }

        return true;
    }

    inline bool Transform::operator!=(const Transform& rhs) const
    {
        return !operator==(rhs);
    }
}
