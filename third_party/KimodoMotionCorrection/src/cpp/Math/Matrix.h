/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Vector.h"
#include "Quaternion.h"

enum class CoordinateSpace : uint8_t
{
    World,
    Local,
};

//
// Matrices are Row-Major
// Multiplication order is right to left
// ObjectWorldTransform = LocalObjectTransform * WorldTransform
//

namespace Math
{
    class alignas(16) Matrix
    {
    public:

        static Matrix const Identity;

    public:

        static Matrix FromRotation(const Quaternion& rotation);
        static Matrix FromTranslation(const Vector& translation);
        static Matrix FromScale(const Vector& scale);
        static Matrix FromUniformScale(float uniformScale);
        static Matrix FromTranslationAndScale(const Vector& translation, const Vector& scale);
        static Matrix FromRotationBetweenVectors(const Vector sourceVector, const Vector targetVector);

    public:

        explicit Matrix();
        explicit Matrix(NoInit_t);
        explicit Matrix(ZeroInit_t);
        explicit Matrix(float v00, float v01, float v02, float v03,
                        float v10, float v11, float v12, float v13,
                        float v20, float v21, float v22, float v23,
                        float v30, float v31, float v32, float v33);
        explicit Matrix(float values[16]);
        explicit Matrix(Vector const& xAxis, Vector const& yAxis, Vector const& zAxis);
        explicit Matrix(Vector const& xAxis, Vector const& yAxis, Vector const& zAxis, Vector const& translation);

        Matrix(const Vector axis, Radians angleRadians);
        Matrix(const AxisAngle axisAngle);

        explicit Matrix(const Quaternion& rotation);
        explicit Matrix(const Quaternion& rotation, const Vector& translation, const Vector& scale = Vector::One);
        explicit Matrix(const Quaternion& rotation, const Vector& translation, float scale = 1.0f);
        explicit Matrix(const EulerAngles& eulerAngles, const Vector translation = Vector::UnitW);

        EulerAngles ToEulerAngles() const;

        float* AsFloatArray();
        const float* AsFloatArray() const;
        const Vector& GetRow(uint32_t row) const;

        const Vector& GetAxisX() const;
        const Vector& GetAxisY() const;
        const Vector& GetAxisZ() const;

        void SetAxisX(const Vector& xAxis);
        void SetAxisY(const Vector& yAxis);
        void SetAxisZ(const Vector& zAxis);

        Float3 GetForwardVector() const;
        Float3 GetRightVector() const;
        Float3 GetUpVector() const;

        Vector GetUnitAxisX() const;
        Vector GetUnitAxisY() const;
        Vector GetUnitAxisZ() const;

        bool IsIdentity() const;
        bool IsOrthogonal() const;
        bool IsOrthonormal() const;

        bool Decompose(Quaternion& outRotation, Vector& outTranslation, Vector& outScale) const;

        Matrix& Transpose();
        Matrix GetTransposed() const;

        Matrix& Invert();
        Matrix GetInverse() const;

        Vector GetDeterminant() const;
        float GetDeterminantAsFloat() const;

        Vector GetTranslation() const;
        const Vector& GetTranslationWithW() const;
        Matrix& SetTranslation(Vector const& v);
        Matrix& SetTranslation(Float3 const& v);
        Matrix& SetTranslation(Float4 const& v);

        Quaternion GetRotation() const;

        Matrix& SetRotation(const Matrix& rotation);
        Matrix& SetRotation(const Quaternion& rotation);

        Matrix& SetRotationMaintainingScale(const Matrix& rotation);
        Matrix& SetRotationMaintainingScale(const Quaternion& rotation);

        Vector GetScale() const;

        Matrix& RemoveScale();
        Matrix& SetScale(const Vector& scale);
        Matrix& SetScale(float uniformScale);

        Matrix& RemoveScaleFast();
        Matrix& SetScaleFast(const Vector& scale);
        Matrix& SetScaleFast(float uniformScale);

        //
        // Operators
        //

        // Applies rotation and scale to a vector and returns a result with the W = 0
        Vector RotateVector(const Vector& vector) const;

        // Applies rotation and scale to a vector and returns a result with the W = 0
        Vector TransformNormal(const Vector& vector) const;

        // Applies the transformation to a given point and ensures the resulting W = 1
        Vector TransformPoint(const Vector& point) const;

        // Applies the transformation to a vector ignoring the W value.
        // Same as TransformPoint with the result W left unchanged
        Vector TransformVector3(const Vector& vector) const;

        // Applies the transformation to a given vector with the result W left unchanged
        Vector TransformVector4(const Vector& vector) const;

        Vector& operator[](uint32_t i);
        const Vector operator[](uint32_t i) const;

        Matrix operator*(const Matrix& rhs) const;
        Matrix& operator*=(const Matrix& rhs);

        Matrix operator*(const Quaternion& rhs) const;
        Matrix operator*=(const Quaternion& rhs);

        bool operator==(const Matrix& rhs) const;

    public:

        union
        {
            Vector      m_rows[4];
            float       m_values[4][4];
        };
    };
}

#include "Matrix.inl"
