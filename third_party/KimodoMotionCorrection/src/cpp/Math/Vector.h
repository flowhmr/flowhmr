/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Compiler.h"

#include "Types.h"
#include "Constants.h"
#include "SIMD.h"

namespace Math
{
    class alignas(16) Vector
    {
    public:

        static Vector const UnitX;
        static Vector const UnitY;
        static Vector const UnitZ;
        static Vector const UnitW;

        static Vector const Origin;
        static Vector const WorldForward;
        static Vector const WorldBackward;
        static Vector const WorldUp;
        static Vector const WorldDown;
        static Vector const WorldLeft;
        static Vector const WorldRight;

        static Vector const NegativeOne;
        static Vector const Zero;
        static Vector const Half;
        static Vector const One;
        static Vector const Epsilon;
        static Vector const LargeEpsilon;
        static Vector const OneMinusEpsilon;
        static Vector const EpsilonMinusOne;
        static Vector const NormalizeCheckThreshold;
        static Vector const Pi;
        static Vector const PiDivTwo;
        static Vector const TwoPi;
        static Vector const OneDivTwoPi;

        static Vector const Select0000;
        static Vector const Select0001;
        static Vector const Select0010;
        static Vector const Select0011;
        static Vector const Select0100;
        static Vector const Select0101;
        static Vector const Select0110;
        static Vector const Select0111;
        static Vector const Select1000;
        static Vector const Select1001;
        static Vector const Select1010;
        static Vector const Select1011;
        static Vector const Select1100;
        static Vector const Select1101;
        static Vector const Select1110;
        static Vector const Select1111;

        static Vector const Infinity;
        static Vector const QNaN;

        static Vector const BoxCorners[8];

        //
        // Utils
        //

        static Vector Cross2(const Vector& v0, const Vector& v1);
        static Vector Cross3(const Vector& v0, const Vector& v1);
        static Vector Dot2(const Vector& v0, const Vector& v1);
        static Vector Dot3(const Vector& v0, const Vector& v1);
        static Vector Dot4(const Vector& v0, const Vector& v1);
        static Vector Average2(const Vector& v0, const Vector& v1);
        static Vector Average3(const Vector& v0, const Vector& v1);
        static Vector Average4(const Vector& v0, const Vector& v1);
        static Vector Min(const Vector& v0, const Vector& v1);
        static Vector Max(const Vector& v0, const Vector& v1);
        static float Min(const Vector& v);
        static float Max(const Vector& v);
        static Vector Clamp(const Vector& v, const Vector& min, const Vector& max);
        static Vector Xor(const Vector& vec0, const Vector& vec1);

        // Add the multiplied results to a vector: ( vec * mul ) + addend
        static Vector MultiplyAdd(const Vector& vec, const Vector& multiplier, const Vector& addend);

        // Subtract a vector from the multiplied result: (vec * mul ) - subtrahend
        static Vector MultiplySubtract(const Vector& vec, const Vector& multiplier, const Vector& subtrahend);

        // Subtract the multiplied result from a vector: minuend - (vec * mul )
        static Vector NegativeMultiplySubtract(const Vector& vec, const Vector& multiplier, const Vector& minuend);

        // Sum up scaled versions of two vectors
        static Vector LinearCombination(const Vector& v0, const Vector& v1, float scale0, float scale1);

        // Linear interpolation of one vector to another
        static Vector Lerp(const Vector& from, const Vector& to, float t);

        // Normalized linear interpolation of one vector to another
        static Vector NLerp(const Vector& from, const Vector& to, float t);

        // Spherical interpolation of one vector to another
        static Vector SLerp(const Vector& from, const Vector& to, float t);

        // Combine the two vectors based on the control: 0 means select from v0, 1 means select from v1. E.G. To select XY from v0 and ZW from v1, control = Vector( 0, 0, 1, 1 )
        static Vector Select(const Vector& v0, const Vector& v1, const Vector& control);

        // Get a permutation of two vectors, each template argument represents the element index to select ( v0: 0-3, v1: 4-7 );
        template<uint32_t PermuteX, uint32_t PermuteY, uint32_t PermuteZ, uint32_t PermuteW>
        static Vector Permute(const Vector& v0, const Vector& v1);

        //
        // Trigonometry
        //

        static Vector Sin(const Vector& vec);
        static Vector Cos(const Vector& vec);
        static Vector Tan(const Vector& vec);
        static Vector ASin(const Vector& vec);
        static Vector ACos(const Vector& vec);
        static Vector ATan(const Vector& vec);
        static Vector ATan2(const Vector& vec0, const Vector& vec1);

        static Vector SinEst(const Vector& vec);
        static Vector CosEst(const Vector& vec);
        static Vector TanEst(const Vector& vec);
        static Vector ASinEst(const Vector& vec);
        static Vector ACosEst(const Vector& vec);
        static Vector ATanEst(const Vector& vec);
        static Vector ATan2Est(const Vector& vec0, const Vector& vec1);

        static void SinCos(Vector& sin, Vector& cos, float angle);
        static void SinCos(Vector& sin, Vector& cos, const Vector& angle);

        static Vector AngleMod2Pi(const Vector& angles);

    public:

        operator __m128& ();
        operator const __m128& () const;

        Vector();
        explicit Vector(Axis axis);
        explicit Vector(ZeroInit_t);
        explicit Vector(float v);
        Vector(__m128 v);
        Vector(float ix, float iy, float iz, float iw = 1.0f);

        Vector(const Float2& v, float iz = 0.0f, float iw = 0.0f);
        Vector(const Float3& v, float iw = 1.0f);
        Vector(const Float4& v);
        Vector(const float* pValues);

        bool IsValid() const;

        void Store(float* pValues) const;
        void StoreFloat(float& value) const;
        void StoreFloat2(Float2& value) const;
        void StoreFloat3(Float3& value) const;
        void StoreFloat4(Float4& value) const;

        float ToFloat() const;
        Float2 ToFloat2() const;
        Float3 ToFloat3() const;
        Float4 ToFloat4() const;

        operator Float2() const;
        operator Float3() const;
        operator Float4() const;

        //
        // Element accessors
        //

        float GetX() const;
        float GetY() const;
        float GetZ() const;
        float GetW() const;

        void SetX(float x);
        void SetY(float y);
        void SetZ(float z);
        void SetW(float w);

        float operator[](uint32_t i) const;

        //
        // W component operations
        //

        bool IsW1() const;
        bool IsW0() const;
        Vector& SetW0();
        Vector& SetW1();
        Vector GetWithW0() const;
        Vector GetWithW1() const;

        //
        // Dimensional Getters
        //

        // Returns only the first two components, z=w=0
        Vector Get2D() const;

        // Returns only the first three components, w = 0
        Vector Get3D() const;

        //
        // Algebraic operators
        //

        Vector operator+(const Vector& v) const;
        Vector& operator+=(const Vector& v);
        Vector operator-(const Vector& v) const;
        Vector& operator-=(const Vector& v);
        Vector operator*(const Vector& v) const;
        Vector& operator*=(const Vector& v);
        Vector operator/(const Vector& v) const;
        Vector& operator/=(const Vector& v);

        Vector operator*(float const f) const;
        Vector& operator*=(float const f);
        Vector operator/(float const f) const;
        Vector& operator/=(float const f);

        Vector operator-() const;

        Vector Orthogonal2D() const;
        Vector Cross2(const Vector& other) const;
        Vector Cross3(const Vector& other) const;
        Vector Dot2(const Vector& other) const;
        Vector Dot3(const Vector& other) const;
        Vector Dot4(const Vector& other) const;
        float GetDot2(const Vector& other) const;
        float GetDot3(const Vector& other) const;
        float GetDot4(const Vector& other) const;

        Vector ScalarProjection(const Vector& other) const;
        float GetScalarProjection(const Vector& other) const;
        Vector VectorProjection(const Vector& other) const;

        //
        // Transformations
        //

        Vector& Invert();
        Vector GetInverse() const;
        Vector GetReciprocal() const;

        Vector& InvertEst();
        Vector GetInverseEst() const;

        Vector& Negate();
        Vector GetNegated() const;

        Vector& Abs();
        Vector GetAbs() const;

        Vector& Sqrt();
        Vector GetSqrt();

        Vector& ReciprocalSqrt();
        Vector GetReciprocalSqrt();

        Vector& EstimatedReciprocalSqrt();
        Vector GetEstimatedReciprocalSqrt();

        Vector& Normalize2();
        Vector& Normalize3();
        Vector& Normalize4();

        Vector GetNormalized2() const;
        Vector GetNormalized3() const;
        Vector GetNormalized4() const;

        Vector& Floor();
        Vector GetFloor() const;
        Vector& Ceil();
        Vector GetCeil() const;
        Vector& Round();
        Vector GetRound() const;

        Vector GetSign() const;

        //
        // Permutations
        //

        Vector GetSplatX() const;
        Vector GetSplatY() const;
        Vector GetSplatZ() const;
        Vector GetSplatW() const;

        // Get a shuffled version of this vector, each argument represents the element index in the original vector
        template<uint32_t xIdx, uint32_t yIdx, uint32_t zIdx, uint32_t wIdx>
        Vector Swizzle() const;

        // Get a shuffled version of this vector, each argument represents the element index in the original vector
        Vector Swizzle(uint32_t xIdx, uint32_t yIdx, uint32_t zIdx, uint32_t wIdx) const;

        // Get a shuffled version of this vector, each argument represents the element index in the original vector
        Vector Shuffle(uint32_t xIdx, uint32_t yIdx, uint32_t zIdx, uint32_t wIdx) const;

        // Get a shuffled version of this vector, each argument represents the element index in the original vector
        template<uint32_t xIdx, uint32_t yIdx, uint32_t zIdx, uint32_t wIdx>
        Vector Shuffle() const;

        //
        // Queries
        //

        Vector Length2() const;
        Vector Length3() const;
        Vector Length4() const;

        float GetLength2() const;
        float GetLength3() const;
        float GetLength4() const;

        Vector InverseLength2() const;
        Vector InverseLength3() const;
        Vector InverseLength4() const;

        float GetInverseLength2() const;
        float GetInverseLength3() const;
        float GetInverseLength4() const;

        Vector LengthSquared2() const;
        Vector LengthSquared3() const;
        Vector LengthSquared4() const;

        float GetLengthSquared2() const;
        float GetLengthSquared3() const;
        float GetLengthSquared4() const;

        Vector Distance2(const Vector& to) const;
        Vector Distance3(const Vector& to) const;
        Vector Distance4(const Vector& to) const;

        float GetDistance2(const Vector& to) const;
        float GetDistance3(const Vector& to) const;
        float GetDistance4(const Vector& to) const;

        Vector DistanceSquared2(const Vector& to) const;
        Vector DistanceSquared3(const Vector& to) const;
        Vector DistanceSquared4(const Vector& to) const;

        float GetDistanceSquared2(const Vector& to) const;
        float GetDistanceSquared3(const Vector& to) const;
        float GetDistanceSquared4(const Vector& to) const;

        bool IsNormalized2() const;
        bool IsNormalized3() const;
        bool IsNormalized4() const;

        // Is this vector within the range [-bounds, bounds]
        Vector InBounds(const Vector& bounds) const;

        bool IsInBounds2(const Vector& bounds) const;
        bool IsInBounds3(const Vector& bounds) const;
        bool IsInBounds4(const Vector& bounds) const;

        Vector Equal(const Vector& v) const;

        bool IsEqual2(const Vector& v) const;
        bool IsEqual3(const Vector& v) const;
        bool IsEqual4(const Vector& v) const;

        Vector NearEqual(const Vector& v, const Vector& epsilon) const;

        bool IsNearEqual2(const Vector& v, float epsilon) const;
        bool IsNearEqual3(const Vector& v, float epsilon) const;
        bool IsNearEqual4(const Vector& v, float epsilon) const;

        bool IsNearEqual2(const Vector& v, const Vector& epsilon = Vector::Epsilon) const;
        bool IsNearEqual3(const Vector& v, const Vector& epsilon = Vector::Epsilon) const;
        bool IsNearEqual4(const Vector& v, const Vector& epsilon = Vector::Epsilon) const;

        Vector GreaterThan(const Vector& v) const;
        bool IsAnyGreaterThan(const Vector& v) const;

        bool IsGreaterThan2(const Vector& v) const;
        bool IsGreaterThan3(const Vector& v) const;
        bool IsGreaterThan4(const Vector& v) const;

        Vector GreaterThanEqual(const Vector& v) const;
        bool IsAnyGreaterThanEqual(const Vector& v) const;

        bool IsGreaterThanEqual2(const Vector& v) const;
        bool IsGreaterThanEqual3(const Vector& v) const;
        bool IsGreaterThanEqual4(const Vector& v) const;

        Vector LessThan(const Vector& v) const;
        bool IsAnyLessThan(const Vector& v) const;

        bool IsLessThan2(const Vector& v) const;
        bool IsLessThan3(const Vector& v) const;
        bool IsLessThan4(const Vector& v) const;

        Vector LessThanEqual(const Vector& v) const;
        bool IsAnyLessThanEqual(const Vector& v) const;

        bool IsLessThanEqual2(const Vector& v) const;
        bool IsLessThanEqual3(const Vector& v) const;
        bool IsLessThanEqual4(const Vector& v) const;

        Vector EqualsZero() const;
        bool IsAnyEqualToZero2() const;
        bool IsAnyEqualToZero3() const;
        bool IsAnyEqualToZero4() const;

        bool IsZero2() const;
        bool IsZero3() const;
        bool IsZero4() const;

        Vector NearEqualsZero(float epsilon = Math::Epsilon) const;

        bool IsNearZero2(float epsilon = Math::Epsilon) const;
        bool IsNearZero3(float epsilon = Math::Epsilon) const;
        bool IsNearZero4(float epsilon = Math::Epsilon) const;

        Vector EqualsInfinity() const;

        bool IsInfinite2() const;
        bool IsInfinite3() const;
        bool IsInfinite4() const;

        Vector EqualsNaN() const;

        bool IsNaN2() const;
        bool IsNaN3() const;
        bool IsNaN4() const;

        bool IsParallelTo(const Vector& v) const;

        void ToDirectionAndLength2(Vector& direction, float& length) const;
        void ToDirectionAndLength3(Vector& direction, float& length) const;

        bool operator==(const Vector& rhs) const;
        bool operator!=(const Vector& rhs) const;

    public:

        __m128 m_data;
    };

    static_assert(sizeof(Vector) == 16, "Vector size must be 16 bytes!");
}

#include "Vector.inl"
