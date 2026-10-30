/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Vector.h"

namespace Math
{
    class alignas(16) Quaternion
    {
    public:

        static Quaternion const Identity;

        // Calculate the rotation required to align the source vector to the target vector (shortest path)
        static Quaternion FromRotationBetweenNormalizedVectors(const Vector& sourceVector, const Vector& targetVector);

        // Calculate the rotation required to align one vector onto another but also taking account a fallback rotation axis for opposite parallel vectors
        static Quaternion FromRotationBetweenNormalizedVectors(const Vector& sourceVector, const Vector& targetVector, const Vector& fallbackRotationAxis);

        // Calculate the rotation required to align the source vector to the target vector (shortest path)
        static Quaternion FromRotationBetweenVectors(const Vector& sourceVector, const Vector& targetVector);

        // Normalized LERP - not accurate - only use for really short distances
        static Quaternion NLerp(const Quaternion& from, const Quaternion& to, float t);

        // Standard and accurate Spherical LERP - based on DirectX Math
        static Quaternion SLerp(const Quaternion& from, const Quaternion& to, float t);

        // Fast approximation of a Spherical LERP - based on "A fast and accurate estimate for SLERP" by David Eberly
        static Quaternion FastSLerp(const Quaternion& from, const Quaternion& to, float t);

        // Spherical quadrangle/cubic interpolation for quaternions
        static Quaternion SQuad(const Quaternion& q0, const Quaternion& q1, const Quaternion& q2, const Quaternion& q3, float t);

        // Calculate the shortest delta quaternion needed to rotate 'from' onto 'to'
        static Quaternion Delta(const Quaternion& from, const Quaternion& to);

        // Simple vector dot product between two quaternions
        static Vector Dot(const Quaternion& q0, const Quaternion& q1);

        // Calculate the angular distance between two quaternions
        static Radians Distance(const Quaternion& q0, const Quaternion& q1);

        // Calculate look rotation given forward and up vectors
        static Quaternion LookRotation(const Vector& forward, const Vector& up);

    public:

        Quaternion() = default;
        explicit Quaternion(NoInit_t);
        explicit Quaternion(IdentityInit_t);
        explicit Quaternion(const Vector& v);
        explicit Quaternion(float ix, float iy, float iz, float iw);
        explicit Quaternion(const Float4& v);

        explicit Quaternion(const Vector& axis, Radians angle);
        explicit Quaternion(AxisAngle axisAngle);

        explicit Quaternion(const EulerAngles& eulerAngles);
        explicit Quaternion(Radians rotX, Radians rotY, Radians rotZ);

        operator __m128& ();
        operator const __m128& () const;

        Float4 ToFloat4() const;
        Vector ToVector() const;

        Vector Length();
        float GetLength() const;

        // Get the angle this rotation represents around the specified axis
        Radians GetAngle() const;

        AxisAngle ToAxisAngle() const;
        EulerAngles ToEulerAngles() const;

        Vector RotateVector(const Vector& vector) const;
        Vector RotateVectorInverse(const Vector& vector) const;

        Quaternion& Conjugate();
        Quaternion GetConjugate() const;

        Quaternion& Negate();
        Quaternion GetNegated() const;

        Quaternion& Invert();
        Quaternion GetInverse() const;

        Quaternion& Normalize();
        Quaternion GetNormalized() const;

        Vector XAxis() const noexcept;
        Vector YAxis() const noexcept;
        Vector ZAxis() const noexcept;

        // Ensure that this rotation is the shortest in terms of the angle (i.e. -5 instead of 355)
        Quaternion& MakeShortestPath();

        // Ensure that this rotation is the shortest in terms of the angle (i.e. -5 instead of 355)
        Quaternion GetShortestPath() const;

        // This function will return the estimated normalized quaternion, this is not super accurate but a lot faster (use with care)
        Quaternion& NormalizeInaccurate();

        // This function will return the estimated normalized quaternion, this is not super accurate but a lot faster (use with care)
        Quaternion GetNormalizedInaccurate() const;

        bool IsNormalized() const;
        bool IsIdentity() const;

        // Concatenate the rotation of this onto rhs and return the result i.e. first rotate by rhs then by this
        // This means order of rotation is right-to-left: child-rotation * parent-rotation
        Quaternion operator*(const Quaternion& rhs) const;
        Quaternion& operator*=(const Quaternion& rhs);

        // Is the distance between this quaternion and another one under the threshold?
        bool IsNearEqual(const Quaternion& rhs, Radians const threshold = Math::DegreesToRadians) const;

        // Exact equality
        bool operator==(const Quaternion& rhs) const;

        // Exact equality
        bool operator!=(const Quaternion& rhs) const;

    private:

        Vector GetSplatW() const;
        float GetW() const;

        Quaternion& operator=(const Vector& v) = delete;

    public:

        __m128 m_data;
    };

    static_assert(sizeof(Vector) == 16, "Quaternion size must be 16 bytes!");
}

#include "Quaternion.inl"
