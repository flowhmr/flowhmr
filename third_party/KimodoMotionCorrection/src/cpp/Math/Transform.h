/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Matrix.h"

namespace Math
{
    //
    // VQS Transform
    //

    class Transform
    {
    public:

        static Transform const Identity;

        static Transform FromRotation(const Quaternion& rotation);
        static Transform FromTranslation(const Vector& translation);
        static Transform FromScale(float uniformScale);
        static Transform FromTranslationAndScale(const Vector& translation, float uniformScale);
        static Transform FromRotationBetweenVectors(const Vector sourceVector, const Vector targetVector);

        // Linearly interpolate between two transforms - uses NLerp for rotations
        static Transform Lerp(const Transform& from, const Transform& to, float t);

        // Spherically interpolate between two transforms - uses SLerp for rotations
        static Transform Slerp(const Transform& from, const Transform& to, float t);

        // Spherically interpolate between two transforms - uses FastSLerp (SLerp approximation) for rotations
        static Transform FastSlerp(const Transform& from, const Transform& to, float t);

        // Calculate a delta transform that you can concatenate to the
        // 'from' transform to get the 'to' transform. Properly handles the non-uniform scaling case.
        static Transform Delta(const Transform& from, const Transform& to);

        // Calculates a delta transform that you can concatenate to the
        // 'from' transform to get the 'to' transform (ignoring scale)
        static Transform DeltaNoScale(const Transform& from, const Transform& to);

        static void DirectlySetRotation(Transform& transform, Quaternion&& rotation);
        static void DirectlySetRotation(Transform& transform, const Quaternion& rotation);
        static void DirectlySetTranslationScale(Transform& transform, Vector&& translationScale);
        static void DirectlySetTranslationScale(Transform& transform, const Vector& translationScale);

    public:

        Transform() = default;

        explicit Transform(NoInit_t);
        explicit Transform(const Matrix& m);
        explicit Transform(
            const Quaternion& rotation,
                const Vector& translation = Vector(0, 0, 0, 0),
                    float scale = 1.0f);
        explicit Transform(const AxisAngle& rotation);

        Matrix ToMatrix() const;
        Matrix ToMatrixNoScale() const;
        EulerAngles ToEulerAngles() const;

        Vector GetAxisX() const;
        Vector GetAxisY() const;
        Vector GetAxisZ() const;

        Vector GetRightVector() const;
        Vector GetForwardVector() const;
        Vector GetUpVector() const;

        bool IsIdentity() const;
        bool IsRigidTransform() const;
        void MakeRigidTransform();

        //
        // Inverse and Deltas
        //

        // Invert this transform.
        // If you want a delta transform that you can
        // concatenate, then you should use the 'Delta' functions
        Transform& Inverse();

        // Get the inverse of this transform.
        // If you want a delta transform that you can
        // concatenate, then you should use the 'Delta' functions
        Transform GetInverse() const;

        // Return the delta required to a given target
        // transform (i.e., what do we need to add to reach that transform)
        Transform GetDeltaToOther(const Transform& targetTransform) const;

        // Return the delta relative from a given a start
        // transform (i.e., how much do we differ from it)
        Transform GetDeltaFromOther(const Transform& startTransform) const;

        //
        // Rotation

        const Quaternion& GetRotation() const;
        void SetRotation(const Quaternion& rotation);
        void AddRotation(const Quaternion& delta);

        //
        // Translation
        //

        // Get the translation for this transform
        // NOTE: you cannot rely on the W value as that will be the scale
        const Vector& GetTranslation() const;

        // Get the translation and scale for this transform
        const Vector& GetTranslationAndScale() const;

        // Set the translation
        void SetTranslation(const Vector& newTranslation);

        // Set the translation and scale simultaneously
        void SetTranslationAndScale(const Vector& newTranslationScale);

        // Add an offset to the current translation
        void AddTranslation(const Vector& translationDelta);

        // Get the translation as a homogeneous coordinates' vector (W=0)
        Vector GetTranslationAsVector() const;

        // Get the translation as a homogeneous coordinates' point (W=1)
        Vector GetTranslationAsPoint() const;

        //
        // Scale
        //

        float GetScale() const;
        Vector GetScaleVector() const;
        Vector GetInverseScaleVector() const;
        void SetScale(float uniformScale);
        bool HasScale() const;
        bool HasNegativeScale() const;

        // This function will sanitize the scale values to remove any
        // trailing values from scale factors i.e. 1.000000012 will be converted to 1
        // This is primarily needed in import steps where scale values
        // might be sampled from curves or have multiple conversions applied resulting in variance.
        void SanitizeScaleValue();

        //
        // Transformations
        //

        Vector TranslateVector(const Vector& vector) const;
        Vector ScaleVector(const Vector& vector) const;
        Vector TransformPoint(const Vector& vector) const;
        Vector TransformPointNoScale(const Vector& vector) const;

        // Rotate a vector (same as TransformVectorNoScale)
        Vector RotateVector(const Vector& vector) const;

        // Rotate a vector (same as TransformVectorNoScale)
        Vector TransformNormal(const Vector& vector) const;

        // Unrotate a vector (same as InverseTransformVectorNoScale)
        Vector InverseRotateVector(const Vector& vector) const;

        // Invert the operation order when doing inverse transformation: first translation then rotation then scale
        Vector InverseTransformPoint(const Vector& point) const;

        // Invert the operation order when doing inverse transformation: first translation then rotation
        Vector InverseTransformPointNoScale(const Vector& point) const;

        // Applies scale and rotation to a vector (no translation)
        Vector TransformVector(const Vector& vector) const;

        // Rotate a vector
        Vector TransformVectorNoScale(const Vector& vector) const;

        // Invert the operation order when performing inverse transformation: first rotation then scale
        Vector InverseTransformVector(const Vector& vector) const;

        // Unrotate a vector
        Vector InverseTransformVectorNoScale(const Vector& vector) const;

        // WARNING: The results from multiplying transforms with shear or skew is ill-defined
        Transform operator*(const Transform& rhs) const;

        // WARNING: The results from multiplying transforms with shear or skew is ill-defined
        Transform& operator*=(const Transform& rhs);

        //
        // Operators
        //

        bool IsNearEqual(
            const Transform& rhs,
                const Radians angleThreshold = Math::DegreesToRadians,
                    float translationScaleThreshold = Math::Epsilon) const;

        // Exact equality
        bool operator==(const Transform& rhs) const;

        bool operator!=(const Transform& rhs) const;

    private:

        Quaternion m_rotation = Quaternion(0, 0, 0, 1);
        Vector m_translationScale = Vector(0, 0, 0, 1);
    };
}

#include "Transform.inl"
