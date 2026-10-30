/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Scalar.h"

enum NoInit_t { NoInit };
enum ZeroInit_t { ZeroInit };
enum IdentityInit_t { IdentityInit };

enum class Axis : uint8_t
{
    X = 0,
    Y,
    Z,
    NegX,
    NegY,
    NegZ
};

struct Float2;
struct Float3;
struct Float4;

struct Int2
{
    static Int2 const Zero;

public:

    inline Int2() {}
    inline Int2( ZeroInit_t ) : m_x( 0 ), m_y( 0 ) {}
    inline Int2( Float2 const& v );
    inline explicit Int2( int32_t v ) : m_x( v ), m_y( v ) {}
    inline explicit Int2( int32_t ix, int32_t iy ) : m_x( ix ), m_y( iy ) {}

    inline bool IsZero() const { return *this == Zero; }

    inline int32_t& operator[]( uint32_t i ) { return ( (int32_t*) this )[i]; }
    inline int32_t const& operator[]( uint32_t i ) const { return ( (int32_t*) this )[i]; }

    inline bool operator==( Int2 const rhs ) const { return m_x == rhs.m_x && m_y == rhs.m_y; }
    inline bool operator!=( Int2 const rhs ) const { return m_x != rhs.m_x || m_y != rhs.m_y; }

    inline Int2 operator+( Int2 const& rhs ) const { return Int2( m_x + rhs.m_x, m_y + rhs.m_y ); }
    inline Int2 operator-( Int2 const& rhs ) const { return Int2( m_x - rhs.m_x, m_y - rhs.m_y ); }
    inline Int2 operator*( Int2 const& rhs ) const { return Int2( m_x * rhs.m_x, m_y * rhs.m_y ); }
    inline Int2 operator/( Int2 const& rhs ) const { return Int2( m_x / rhs.m_x, m_y / rhs.m_y ); }

    inline Int2& operator+=( int32_t const& rhs ) { m_x += rhs; m_y += rhs; return *this; }
    inline Int2& operator-=( int32_t const& rhs ) { m_x -= rhs; m_y -= rhs; return *this; }
    inline Int2& operator*=( int32_t const& rhs ) { m_x *= rhs; m_y *= rhs; return *this; }
    inline Int2& operator/=( int32_t const& rhs ) { m_x /= rhs; m_y /= rhs; return *this; }

    // Component wise operation
    inline Int2 operator+( int32_t const& rhs ) const { return Int2( m_x + rhs, m_y + rhs ); }
    inline Int2 operator-( int32_t const& rhs ) const { return Int2( m_x - rhs, m_y - rhs ); }
    inline Int2 operator*( int32_t const& rhs ) const { return Int2( m_x * rhs, m_y * rhs ); }
    inline Int2 operator/( int32_t const& rhs ) const { return Int2( m_x / rhs, m_y / rhs ); }

    inline Int2& operator+=( Int2 const& rhs ) { m_x += rhs.m_x; m_y += rhs.m_y; return *this; }
    inline Int2& operator-=( Int2 const& rhs ) { m_x -= rhs.m_x; m_y -= rhs.m_y; return *this; }
    inline Int2& operator*=( Int2 const& rhs ) { m_x *= rhs.m_x; m_y *= rhs.m_y; return *this; }
    inline Int2& operator/=( Int2 const& rhs ) { m_x /= rhs.m_x; m_y /= rhs.m_y; return *this; }

public:

    int32_t m_x, m_y;
};

struct Int3
{
    static Int3 const Zero;

public:

    inline Int3() {}
    inline Int3( ZeroInit_t ) : m_x( 0 ), m_y( 0 ), m_z( 0 ) {}
    inline Int3( Float3 const& v );
    inline explicit Int3( int32_t v ) : m_x( v ), m_y( v ), m_z( v ) {}
    inline explicit Int3( int32_t ix, int32_t iy, int32_t iz ) : m_x( ix ), m_y( iy ), m_z( iz ) {}

    inline bool IsZero() const { return *this == Zero; }

    inline int32_t& operator[]( uint32_t i ) { return ( (int32_t*) this )[i]; }
    inline int32_t const& operator[]( uint32_t i ) const { return ( (int32_t*) this )[i]; }

    inline bool operator==( Int3 const rhs ) const { return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z; }
    inline bool operator!=( Int3 const rhs ) const { return m_x != rhs.m_x || m_y != rhs.m_y || m_z != rhs.m_z; }

    inline Int3 operator+( Int3 const& rhs ) const { return Int3( m_x + rhs.m_x, m_y + rhs.m_y, m_z + rhs.m_z ); }
    inline Int3 operator-( Int3 const& rhs ) const { return Int3( m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z ); }
    inline Int3 operator*( Int3 const& rhs ) const { return Int3( m_x * rhs.m_x, m_y * rhs.m_y, m_z * rhs.m_z ); }
    inline Int3 operator/( Int3 const& rhs ) const { return Int3( m_x / rhs.m_x, m_y / rhs.m_y, m_z / rhs.m_z ); }

    inline Int3& operator+=( int32_t const& rhs ) { m_x += rhs; m_y += rhs; m_z += rhs; return *this; }
    inline Int3& operator-=( int32_t const& rhs ) { m_x -= rhs; m_y -= rhs; m_z -= rhs; return *this; }
    inline Int3& operator*=( int32_t const& rhs ) { m_x *= rhs; m_y *= rhs; m_z *= rhs; return *this; }
    inline Int3& operator/=( int32_t const& rhs ) { m_x /= rhs; m_y /= rhs; m_z /= rhs; return *this; }

    // Component wise operation
    inline Int3 operator+( int32_t const& rhs ) const { return Int3( m_x + rhs, m_y + rhs, m_z + rhs ); }
    inline Int3 operator-( int32_t const& rhs ) const { return Int3( m_x - rhs, m_y - rhs, m_z - rhs ); }
    inline Int3 operator*( int32_t const& rhs ) const { return Int3( m_x * rhs, m_y * rhs, m_z * rhs ); }
    inline Int3 operator/( int32_t const& rhs ) const { return Int3( m_x / rhs, m_y / rhs, m_z / rhs ); }

    inline Int3& operator+=( Int3 const& rhs ) { m_x += rhs.m_x; m_y += rhs.m_y; m_z += rhs.m_z; return *this; }
    inline Int3& operator-=( Int3 const& rhs ) { m_x -= rhs.m_x; m_y -= rhs.m_y; m_z -= rhs.m_z; return *this; }
    inline Int3& operator*=( Int3 const& rhs ) { m_x *= rhs.m_x; m_y *= rhs.m_y; m_z *= rhs.m_z; return *this; }
    inline Int3& operator/=( Int3 const& rhs ) { m_x /= rhs.m_x; m_y /= rhs.m_y; m_z /= rhs.m_z; return *this; }

public:

    int32_t m_x, m_y, m_z;
};

struct Int4
{
    static Int4 const Zero;
    static Int4 const MinusOne;

public:

    inline Int4() {}
    inline Int4( ZeroInit_t ) : m_x( 0 ), m_y( 0 ), m_z( 0 ), m_w( 0 ) {}
    inline explicit Int4( int32_t v ) : m_x( v ), m_y( v ), m_z( v ), m_w( v ) {}
    inline explicit Int4( int32_t ix, int32_t iy, int32_t iz, int32_t iw ) : m_x( ix ), m_y( iy ), m_z( iz ), m_w( iw ) {}

    inline bool IsZero() const { return *this == Zero; }

    inline int32_t& operator[]( uint32_t i ) { return ( (int32_t*) this )[i]; }
    inline int32_t const& operator[]( uint32_t i ) const { return ( (int32_t*) this )[i]; }

    inline bool operator==( Int4 const rhs ) const { return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z && m_w == rhs.m_w; }
    inline bool operator!=( Int4 const rhs ) const { return m_x != rhs.m_x || m_y != rhs.m_y || m_z != rhs.m_z || m_w != rhs.m_w; }

    inline Int4 operator+( int32_t const& rhs ) const { return Int4( m_x + rhs, m_y + rhs, m_z + rhs, m_w + rhs ); }
    inline Int4 operator-( int32_t const& rhs ) const { return Int4( m_x - rhs, m_y - rhs, m_z - rhs, m_w - rhs ); }
    inline Int4 operator*( int32_t const& rhs ) const { return Int4( m_x * rhs, m_y * rhs, m_z * rhs, m_w * rhs ); }
    inline Int4 operator/( int32_t const& rhs ) const { return Int4( m_x / rhs, m_y / rhs, m_z / rhs, m_w / rhs ); }

    inline Int4& operator+=( int32_t const& rhs ) { m_x += rhs; m_y += rhs; m_z += rhs; m_w += rhs; return *this; }
    inline Int4& operator-=( int32_t const& rhs ) { m_x -= rhs; m_y -= rhs; m_z -= rhs; m_w -= rhs; return *this; }
    inline Int4& operator*=( int32_t const& rhs ) { m_x *= rhs; m_y *= rhs; m_z *= rhs; m_w *= rhs; return *this; }
    inline Int4& operator/=( int32_t const& rhs ) { m_x /= rhs; m_y /= rhs; m_z /= rhs; m_w /= rhs; return *this; }

    // Component wise operation
    inline Int4 operator+( Int4 const& rhs ) const { return Int4( m_x + rhs.m_x, m_y + rhs.m_y, m_z + rhs.m_z, m_w + rhs.m_w ); }
    inline Int4 operator-( Int4 const& rhs ) const { return Int4( m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z, m_w - rhs.m_w ); }
    inline Int4 operator*( Int4 const& rhs ) const { return Int4( m_x * rhs.m_x, m_y * rhs.m_y, m_z * rhs.m_z, m_w * rhs.m_w ); }
    inline Int4 operator/( Int4 const& rhs ) const { return Int4( m_x / rhs.m_x, m_y / rhs.m_y, m_z / rhs.m_z, m_w / rhs.m_w ); }

    inline Int4& operator+=( Int4 const& rhs ) { m_x += rhs.m_x; m_y += rhs.m_y; m_z += rhs.m_z; m_w += rhs.m_w; return *this; }
    inline Int4& operator-=( Int4 const& rhs ) { m_x -= rhs.m_x; m_y -= rhs.m_y; m_z -= rhs.m_z; m_w -= rhs.m_w; return *this; }
    inline Int4& operator*=( Int4 const& rhs ) { m_x *= rhs.m_x; m_y *= rhs.m_y; m_z *= rhs.m_z; m_w *= rhs.m_w; return *this; }
    inline Int4& operator/=( Int4 const& rhs ) { m_x /= rhs.m_x; m_y /= rhs.m_y; m_z /= rhs.m_z; m_w /= rhs.m_w; return *this; }

public:

    int32_t m_x, m_y, m_z, m_w;
};

struct Float2
{
    static Float2 const Zero;
    static Float2 const One;
    static Float2 const UnitX;
    static Float2 const UnitY;

public:

    inline Float2() {}
    FORCE_INLINE Float2( ZeroInit_t ) : m_x( 0 ), m_y( 0 ) {}
    FORCE_INLINE explicit Float2( float v ) : m_x( v ), m_y( v ) {}
    FORCE_INLINE explicit Float2( float ix, float iy ) : m_x( ix ), m_y( iy ) {}
    FORCE_INLINE explicit Float2( int32_t ix, int32_t iy ) : m_x( (float) ix ), m_y( (float) iy ) {}
    inline explicit Float2( Int2 const& v ) : m_x( (float) v.m_x ), m_y( (float) v.m_y ) {}
    inline explicit Float2( Float3 const& v );
    inline explicit Float2( Float4 const& v );

    inline bool IsZero() const { return *this == Zero; }

    inline float& operator[]( uint32_t i ) { return ( (float*) this )[i]; }
    inline float const& operator[]( uint32_t i ) const { return ( (float*) this )[i]; }

    FORCE_INLINE Float2 operator-() const { return Float2( -m_x, -m_y ); }

    inline bool operator==( Float2 const rhs ) const { return m_x == rhs.m_x && m_y == rhs.m_y; }
    inline bool operator!=( Float2 const rhs ) const { return m_x != rhs.m_x || m_y != rhs.m_y; }

    inline Float2 operator+( Float2 const& rhs ) const { return Float2( m_x + rhs.m_x, m_y + rhs.m_y ); }
    inline Float2 operator-( Float2 const& rhs ) const { return Float2( m_x - rhs.m_x, m_y - rhs.m_y ); }
    inline Float2 operator*( Float2 const& rhs ) const { return Float2( m_x * rhs.m_x, m_y * rhs.m_y ); }
    inline Float2 operator/( Float2 const& rhs ) const { return Float2( m_x / rhs.m_x, m_y / rhs.m_y ); }

    inline Float2 operator+( float const& rhs ) const { return Float2( m_x + rhs, m_y + rhs ); }
    inline Float2 operator-( float const& rhs ) const { return Float2( m_x - rhs, m_y - rhs ); }
    inline Float2 operator*( float const& rhs ) const { return Float2( m_x * rhs, m_y * rhs ); }
    inline Float2 operator/( float const& rhs ) const { return Float2( m_x / rhs, m_y / rhs ); }

    inline Float2& operator+=( Float2 const& rhs ) { m_x += rhs.m_x; m_y += rhs.m_y; return *this; }
    inline Float2& operator-=( Float2 const& rhs ) { m_x -= rhs.m_x; m_y -= rhs.m_y; return *this; }
    inline Float2& operator*=( Float2 const& rhs ) { m_x *= rhs.m_x; m_y *= rhs.m_y; return *this; }
    inline Float2& operator/=( Float2 const& rhs ) { m_x /= rhs.m_x; m_y /= rhs.m_y; return *this; }

    inline Float2& operator+=( float const& rhs ) { m_x += rhs; m_y += rhs; return *this; }
    inline Float2& operator-=( float const& rhs ) { m_x -= rhs; m_y -= rhs; return *this; }
    inline Float2& operator*=( float const& rhs ) { m_x *= rhs; m_y *= rhs; return *this; }
    inline Float2& operator/=( float const& rhs ) { m_x /= rhs; m_y /= rhs; return *this; }

    float m_x, m_y;
};

struct Float3
{
    static Float3 const Zero;
    static Float3 const One;
    static Float3 const UnitX;
    static Float3 const UnitY;
    static Float3 const UnitZ;

    static Float3 const WorldForward;
    static Float3 const WorldUp;
    static Float3 const WorldRight;

public:

    inline Float3() {}
    FORCE_INLINE Float3( ZeroInit_t ) : m_x( 0 ), m_y( 0 ), m_z( 0 ) {}
    FORCE_INLINE explicit Float3( float v ) : m_x( v ), m_y( v ), m_z( v ) {}
    FORCE_INLINE explicit Float3( float ix, float iy, float iz ) : m_x( ix ), m_y( iy ), m_z( iz ) {}
    inline explicit Float3( Float2 const& v, float iz = 0.0f ) : m_x( v.m_x ), m_y( v.m_y ), m_z( iz ) {}
    inline explicit Float3( Float4 const& v );

    inline bool IsZero() const { return *this == Zero; }

    inline float& operator[]( uint32_t i ) { return ( (float*) this )[i]; }
    inline float const& operator[]( uint32_t i ) const { return ( (float*) this )[i]; }

    FORCE_INLINE Float3 operator-() const { return Float3( -m_x, -m_y, -m_z ); }

    inline bool operator==( Float3 const rhs ) const { return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z; }
    inline bool operator!=( Float3 const rhs ) const { return m_x != rhs.m_x || m_y != rhs.m_y || m_z != rhs.m_z; }

    inline operator Float2() const { return Float2( m_x, m_y ); }

    inline Float3 operator+( Float3 const& rhs ) const { return Float3( m_x + rhs.m_x, m_y + rhs.m_y, m_z + rhs.m_z ); }
    inline Float3 operator-( Float3 const& rhs ) const { return Float3( m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z ); }
    inline Float3 operator*( Float3 const& rhs ) const { return Float3( m_x * rhs.m_x, m_y * rhs.m_y, m_z * rhs.m_z ); }
    inline Float3 operator/( Float3 const& rhs ) const { return Float3( m_x / rhs.m_x, m_y / rhs.m_y, m_z / rhs.m_z ); }

    inline Float3 operator+( float const& rhs ) const { return Float3( m_x + rhs, m_y + rhs, m_z + rhs ); }
    inline Float3 operator-( float const& rhs ) const { return Float3( m_x - rhs, m_y - rhs, m_z - rhs ); }
    inline Float3 operator*( float const& rhs ) const { return Float3( m_x * rhs, m_y * rhs, m_z * rhs ); }
    inline Float3 operator/( float const& rhs ) const { return Float3( m_x / rhs, m_y / rhs, m_z / rhs ); }

    inline Float3& operator+=( Float3 const& rhs ) { m_x += rhs.m_x; m_y += rhs.m_y; m_z += rhs.m_z; return *this; }
    inline Float3& operator-=( Float3 const& rhs ) { m_x -= rhs.m_x; m_y -= rhs.m_y; m_z -= rhs.m_z; return *this; }
    inline Float3& operator*=( Float3 const& rhs ) { m_x *= rhs.m_x; m_y *= rhs.m_y; m_z *= rhs.m_z; return *this; }
    inline Float3& operator/=( Float3 const& rhs ) { m_x /= rhs.m_x; m_y /= rhs.m_y; m_z /= rhs.m_z; return *this; }

    inline Float3& operator+=( float const& rhs ) { m_x += rhs; m_y += rhs; m_z += rhs; return *this; }
    inline Float3& operator-=( float const& rhs ) { m_x -= rhs; m_y -= rhs; m_z -= rhs; return *this; }
    inline Float3& operator*=( float const& rhs ) { m_x *= rhs; m_y *= rhs; m_z *= rhs; return *this; }
    inline Float3& operator/=( float const& rhs ) { m_x /= rhs; m_y /= rhs; m_z /= rhs; return *this; }

    float m_x, m_y, m_z;
};

struct Float4
{
    static Float4 const Zero;
    static Float4 const One;
    static Float4 const UnitX;
    static Float4 const UnitY;
    static Float4 const UnitZ;
    static Float4 const UnitW;

    static Float4 const WorldForward;
    static Float4 const WorldUp;
    static Float4 const WorldRight;

public:

    Float4() {}
    FORCE_INLINE Float4( ZeroInit_t ) : m_x( 0 ), m_y( 0 ), m_z( 0 ), m_w( 0 ) {}
    FORCE_INLINE explicit Float4( float v ) : m_x( v ), m_y( v ), m_z( v ), m_w( v ) {}
    FORCE_INLINE explicit Float4( float ix, float iy, float iz, float iw ) : m_x( ix ), m_y( iy ), m_z( iz ), m_w( iw ) {}
    explicit Float4( Float2 const& v, float iz = 0.0f, float iw = 0.0f ) : m_x( v.m_x ), m_y( v.m_y ), m_z( iz ), m_w( iw ) {}
    explicit Float4( Float3 const& v, float iw = 0.0f ) : m_x( v.m_x ), m_y( v.m_y ), m_z( v.m_z ), m_w( iw ) {}

    inline bool IsZero() const { return *this == Zero; }

    float& operator[]( uint32_t i ) { return ( (float*) this )[i]; }
    float const& operator[]( uint32_t i ) const { return ( (float*) this )[i]; }

    FORCE_INLINE Float4 operator-() const { return Float4( -m_x, -m_y, -m_z, -m_w ); }

    bool operator==( Float4 const rhs ) const { return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z && m_w == rhs.m_w; }
    bool operator!=( Float4 const rhs ) const { return m_x != rhs.m_x || m_y != rhs.m_y || m_z != rhs.m_z || m_w != rhs.m_w; }

    inline operator Float2() const { return Float2( m_x, m_y ); }
    inline operator Float3() const { return Float3( m_x, m_y, m_z ); }

    inline Float4 operator+( Float4 const& rhs ) const { return Float4( m_x + rhs.m_x, m_y + rhs.m_y, m_z + rhs.m_z, m_w + rhs.m_w ); }
    inline Float4 operator-( Float4 const& rhs ) const { return Float4( m_x - rhs.m_x, m_y - rhs.m_y, m_z - rhs.m_z, m_w - rhs.m_w ); }
    inline Float4 operator*( Float4 const& rhs ) const { return Float4( m_x * rhs.m_x, m_y * rhs.m_y, m_z * rhs.m_z, m_w * rhs.m_w ); }
    inline Float4 operator/( Float4 const& rhs ) const { return Float4( m_x / rhs.m_x, m_y / rhs.m_y, m_z / rhs.m_z, m_w / rhs.m_w ); }

    inline Float4 operator+( float const& rhs ) const { return Float4( m_x + rhs, m_y + rhs, m_z + rhs, m_w + rhs ); }
    inline Float4 operator-( float const& rhs ) const { return Float4( m_x - rhs, m_y - rhs, m_z - rhs, m_w - rhs ); }
    inline Float4 operator*( float const& rhs ) const { return Float4( m_x * rhs, m_y * rhs, m_z * rhs, m_w * rhs ); }
    inline Float4 operator/( float const& rhs ) const { return Float4( m_x / rhs, m_y / rhs, m_z / rhs, m_w / rhs ); }

    inline Float4& operator+=( Float4 const& rhs ) { m_x += rhs.m_x; m_y += rhs.m_y; m_z += rhs.m_z; m_w += rhs.m_w; return *this; }
    inline Float4& operator-=( Float4 const& rhs ) { m_x -= rhs.m_x; m_y -= rhs.m_y; m_z -= rhs.m_z; m_w -= rhs.m_w; return *this; }
    inline Float4& operator*=( Float4 const& rhs ) { m_x *= rhs.m_x; m_y *= rhs.m_y; m_z *= rhs.m_z; m_w *= rhs.m_w; return *this; }
    inline Float4& operator/=( Float4 const& rhs ) { m_x /= rhs.m_x; m_y /= rhs.m_y; m_z /= rhs.m_z; m_w /= rhs.m_w; return *this; }

    inline Float4& operator+=( float const& rhs ) { m_x += rhs; m_y += rhs; m_z += rhs; m_w += rhs; return *this; }
    inline Float4& operator-=( float const& rhs ) { m_x -= rhs; m_y -= rhs; m_z -= rhs; m_w -= rhs; return *this; }
    inline Float4& operator*=( float const& rhs ) { m_x *= rhs; m_y *= rhs; m_z *= rhs; m_w *= rhs; return *this; }
    inline Float4& operator/=( float const& rhs ) { m_x /= rhs; m_y /= rhs; m_z /= rhs; m_w /= rhs; return *this; }

    float m_x, m_y, m_z, m_w;
};

inline Int2::Int2( Float2 const& v )
    : m_x( (int32_t) v.m_x )
    , m_y( (int32_t) v.m_y )
{
}

inline Int3::Int3( Float3 const& v )
    : m_x( (int32_t) v.m_x )
    , m_y( (int32_t) v.m_y )
    , m_z( (int32_t) v.m_z )
{
}

inline Float2::Float2( Float3 const& v )
    : m_x( v.m_x )
    , m_y( v.m_y )
{
}

inline Float2::Float2( Float4 const& v )
    : m_x( v.m_x )
    , m_y( v.m_y )
{
}

inline Float3::Float3( Float4 const& v )
    : m_x( v.m_x )
    , m_y( v.m_y )
    , m_z( v.m_z )
{
}

struct Radians;
struct Degrees;

struct Degrees
{
public:

    inline Degrees() = default;
    inline Degrees( float degrees ) : m_value( degrees ) {}
    inline explicit Degrees( Radians const& radians );

    FORCE_INLINE explicit operator float() const { return m_value; }
    FORCE_INLINE operator Radians() const;
    FORCE_INLINE float ToFloat() const { return m_value; }
    FORCE_INLINE Radians ToRadians() const;

    inline Degrees operator-() const { return Degrees( -m_value ); }

    inline Degrees operator+( Degrees const& rhs ) const { return Degrees( m_value + rhs.m_value ); }
    inline Degrees operator-( Degrees const& rhs ) const { return Degrees( m_value - rhs.m_value ); }
    inline Degrees operator*( Degrees const& rhs ) const { return Degrees( m_value * rhs.m_value ); }
    inline Degrees operator/( Degrees const& rhs ) const { return Degrees( m_value / rhs.m_value ); }

    inline Degrees& operator+=( Degrees const& rhs ) { m_value += rhs.m_value; return *this; }
    inline Degrees& operator-=( Degrees const& rhs ) { m_value -= rhs.m_value; return *this; }
    inline Degrees& operator*=( Degrees const& rhs ) { m_value *= rhs.m_value; return *this; }
    inline Degrees& operator/=( Degrees const& rhs ) { m_value /= rhs.m_value; return *this; }

    inline Degrees operator+( float const& rhs ) const { return Degrees( m_value + rhs ); }
    inline Degrees operator-( float const& rhs ) const { return Degrees( m_value - rhs ); }
    inline Degrees operator*( float const& rhs ) const { return Degrees( m_value * rhs ); }
    inline Degrees operator/( float const& rhs ) const { return Degrees( m_value / rhs ); }

    inline Degrees& operator+=( float const& rhs ) { m_value += rhs; return *this; }
    inline Degrees& operator-=( float const& rhs ) { m_value -= rhs; return *this; }
    inline Degrees& operator*=( float const& rhs ) { m_value *= rhs; return *this; }
    inline Degrees& operator/=( float const& rhs ) { m_value /= rhs; return *this; }

    inline Degrees operator+( int32_t const& rhs ) const { return Degrees( m_value + rhs ); }
    inline Degrees operator-( int32_t const& rhs ) const { return Degrees( m_value - rhs ); }
    inline Degrees operator*( int32_t const& rhs ) const { return Degrees( m_value * rhs ); }
    inline Degrees operator/( int32_t const& rhs ) const { return Degrees( m_value / rhs ); }

    inline Degrees& operator+=( int32_t const& rhs ) { m_value += rhs; return *this; }
    inline Degrees& operator-=( int32_t const& rhs ) { m_value -= rhs; return *this; }
    inline Degrees& operator*=( int32_t const& rhs ) { m_value *= rhs; return *this; }
    inline Degrees& operator/=( int32_t const& rhs ) { m_value /= rhs; return *this; }

    inline Degrees operator+( uint32_t const& rhs ) const { return Degrees( m_value + rhs ); }
    inline Degrees operator-( uint32_t const& rhs ) const { return Degrees( m_value - rhs ); }
    inline Degrees operator*( uint32_t const& rhs ) const { return Degrees( m_value * rhs ); }
    inline Degrees operator/( uint32_t const& rhs ) const { return Degrees( m_value / rhs ); }

    inline Degrees& operator+=( uint32_t const& rhs ) { m_value += rhs; return *this; }
    inline Degrees& operator-=( uint32_t const& rhs ) { m_value -= rhs; return *this; }
    inline Degrees& operator*=( uint32_t const& rhs ) { m_value *= rhs; return *this; }
    inline Degrees& operator/=( uint32_t const& rhs ) { m_value /= rhs; return *this; }

    inline bool operator>( float const& rhs ) const { return m_value > rhs; };
    inline bool operator<( float const& rhs ) const { return m_value < rhs; }
    inline bool operator>=( float const& rhs ) const { return m_value >= rhs; }
    inline bool operator<=( float const& rhs ) const { return m_value <= rhs; }

    inline bool operator>( Degrees const& rhs ) const { return m_value > rhs.m_value; }
    inline bool operator<( Degrees const& rhs ) const { return m_value < rhs.m_value; }
    inline bool operator>=( Degrees const& rhs ) const { return m_value >= rhs.m_value; }
    inline bool operator<=( Degrees const& rhs ) const { return m_value <= rhs.m_value; }

    inline bool operator>( Radians const& rhs ) const;
    inline bool operator<( Radians const& rhs ) const;
    inline bool operator>=( Radians const& rhs ) const;
    inline bool operator<=( Radians const& rhs ) const;

    inline bool operator==( float const& v ) const { return Math::IsNearEqual( m_value, v ); }
    inline bool operator!=( float const& v ) const { return !Math::IsNearEqual( m_value, v ); }

    inline bool operator==( Degrees const& rhs ) const  { return m_value == rhs.m_value; }
    inline bool operator!=( Degrees const& rhs ) const  { return m_value != rhs.m_value; }

    inline bool operator==( Radians const& rhs ) const;
    inline bool operator!=( Radians const& rhs ) const;

    inline void Clamp( Degrees min, Degrees max )
    {
        m_value = Math::Clamp( m_value, min.m_value, max.m_value );
    }

    // Clamps between -360 and 360
    inline void Clamp360()
    {
        m_value -= ( int32_t( m_value / 360.0f ) * 360.0f );
    }

    // Clamps between -360 and 360
    inline Degrees GetClamped360() const
    {
        Degrees d( m_value );
        d.Clamp360();
        return d;
    }

    // Clamps to -180 to 180
    inline void Clamp180()
    {
        Clamp360();

        float delta = 180 - Math::Abs( m_value );
        if ( delta < 0 )
        {
            delta += 180;
            m_value = ( m_value < 0 ) ? delta : -delta;
        }
    }

    // Clamps to -180 to 180
    inline Degrees GetClamped180() const
    {
        Degrees r( m_value );
        r.Clamp180();
        return r;
    }

    // Clamps between 0 to 360
    inline Degrees& ClampPositive360()
    {
        Clamp360();
        if ( m_value < 0 )
        {
            m_value += 360;
        }
        return *this;
    }

    // Clamps between 0 to 360
    inline Degrees GetClampedPositive360() const
    {
        Degrees d( m_value );
        d.ClampPositive360();
        return d;
    }

private:

    float m_value = 0;
};

struct Radians
{
    static Radians const Pi;
    static Radians const TwoPi;
    static Radians const OneDivPi;
    static Radians const OneDivTwoPi;
    static Radians const PiDivTwo;
    static Radians const PiDivFour;

public:

    inline Radians() = default;
    inline Radians( float radians ) : m_value( radians ) {}
    inline explicit Radians( Degrees const& degrees );

    FORCE_INLINE explicit operator float() const { return m_value; }
    FORCE_INLINE operator Degrees() const { return ToDegrees(); }
    FORCE_INLINE float ToFloat() const { return m_value; }
    FORCE_INLINE Degrees ToDegrees() const { return Degrees( m_value * Math::RadiansToDegrees ); }

    inline Radians operator-() const { return Radians( -m_value ); }

    inline Radians operator+( Radians const& rhs ) const { return Radians( m_value + rhs.m_value ); }
    inline Radians operator-( Radians const& rhs ) const { return Radians( m_value - rhs.m_value ); }
    inline Radians operator*( Radians const& rhs ) const { return Radians( m_value * rhs.m_value ); }
    inline Radians operator/( Radians const& rhs ) const { return Radians( m_value / rhs.m_value ); }

    inline Radians& operator+=( Radians const& rhs ) { m_value += rhs.m_value; return *this; }
    inline Radians& operator-=( Radians const& rhs ) { m_value -= rhs.m_value; return *this; }
    inline Radians& operator*=( Radians const& rhs ) { m_value *= rhs.m_value; return *this; }
    inline Radians& operator/=( Radians const& rhs ) { m_value /= rhs.m_value; return *this; }

    inline Radians operator+( float const& rhs ) const { return Radians( m_value + rhs ); }
    inline Radians operator-( float const& rhs ) const { return Radians( m_value - rhs ); }
    inline Radians operator*( float const& rhs ) const { return Radians( m_value * rhs ); }
    inline Radians operator/( float const& rhs ) const { return Radians( m_value / rhs ); }

    inline Radians& operator+=( float const& rhs ) { m_value += rhs; return *this; }
    inline Radians& operator-=( float const& rhs ) { m_value -= rhs; return *this; }
    inline Radians& operator*=( float const& rhs ) { m_value *= rhs; return *this; }
    inline Radians& operator/=( float const& rhs ) { m_value /= rhs; return *this; }

    inline Radians operator+( int32_t const& rhs ) const { return Radians( m_value + rhs ); }
    inline Radians operator-( int32_t const& rhs ) const { return Radians( m_value - rhs ); }
    inline Radians operator*( int32_t const& rhs ) const { return Radians( m_value * rhs ); }
    inline Radians operator/( int32_t const& rhs ) const { return Radians( m_value / rhs ); }

    inline Radians& operator+=( int32_t const& rhs ) { m_value += rhs; return *this; }
    inline Radians& operator-=( int32_t const& rhs ) { m_value -= rhs; return *this; }
    inline Radians& operator*=( int32_t const& rhs ) { m_value *= rhs; return *this; }
    inline Radians& operator/=( int32_t const& rhs ) { m_value /= rhs; return *this; }

    inline Radians operator+( uint32_t const& rhs ) const { return Radians( m_value + rhs ); }
    inline Radians operator-( uint32_t const& rhs ) const { return Radians( m_value - rhs ); }
    inline Radians operator*( uint32_t const& rhs ) const { return Radians( m_value * rhs ); }
    inline Radians operator/( uint32_t const& rhs ) const { return Radians( m_value / rhs ); }

    inline Radians& operator+=( uint32_t const& rhs ) { m_value += rhs; return *this; }
    inline Radians& operator-=( uint32_t const& rhs ) { m_value -= rhs; return *this; }
    inline Radians& operator*=( uint32_t const& rhs ) { m_value *= rhs; return *this; }
    inline Radians& operator/=( uint32_t const& rhs ) { m_value /= rhs; return *this; }

    inline bool operator>( float const& rhs ) const { return m_value > rhs; };
    inline bool operator<( float const& rhs ) const { return m_value < rhs; }
    inline bool operator>=( float const& rhs ) const { return m_value >= rhs; }
    inline bool operator<=( float const& rhs ) const { return m_value <= rhs; }

    inline bool operator>( Radians const& rhs ) const { return m_value > rhs.m_value; }
    inline bool operator<( Radians const& rhs ) const { return m_value < rhs.m_value; }
    inline bool operator>=( Radians const& rhs ) const { return m_value >= rhs.m_value; }
    inline bool operator<=( Radians const& rhs ) const { return m_value <= rhs.m_value; }

    inline bool operator>( Degrees const& rhs ) const;
    inline bool operator<( Degrees const& rhs ) const;
    inline bool operator>=( Degrees const& rhs ) const;
    inline bool operator<=( Degrees const& rhs ) const;

    inline bool operator==( float const& v ) const { return Math::IsNearEqual( m_value, v ); }
    inline bool operator!=( float const& v ) const { return !Math::IsNearEqual( m_value, v ); }

    inline bool operator==( Radians const& rhs ) const { return m_value == rhs.m_value; }
    inline bool operator!=( Radians const& rhs ) const { return m_value != rhs.m_value; }

    inline bool operator==( Degrees const& rhs ) const;
    inline bool operator!=( Degrees const& rhs ) const;

    inline void Clamp( Radians min, Radians max )
    {
        m_value = Math::Clamp( m_value, min.m_value, max.m_value );
    }

    // Clamps between -2Pi to 2Pi
    inline void Clamp360()
    {
        m_value -= int32_t( m_value / Math::TwoPi ) * Math::TwoPi;
    }

    // Clamps between -2Pi to 2Pi
    inline Radians GetClamped360() const
    {
        Radians r( m_value );
        r.Clamp360();
        return r;
    }

    // Clamps between 0 to 2Pi
    inline void ClampPositive360()
    {
        Clamp360();
        if( m_value < 0 )
        {
            m_value += Math::TwoPi;
        }
    }

    // Clamps between 0 to 2Pi
    inline Radians GetClampedToPositive360() const
    {
        Radians r( m_value );
        r.ClampPositive360();
        return r;
    }

    // Clamps to -Pi to Pi
    inline void Clamp180()
    {
        Clamp360();

        float delta = Math::Pi - Math::Abs( m_value );
        if ( delta < 0 )
        {
            delta += Math::Pi;
            m_value = ( m_value < 0 ) ? delta : -delta;
        }
    }

    // Clamps to -Pi to Pi
    inline Radians GetClamped180() const
    {
        Radians r( m_value );
        r.Clamp180();
        return r;
    }

    // Inverts angle between [0;2Pi] and [-2Pi;0]
    inline void Invert()
    {
        Clamp360();
        float const delta = Math::TwoPi - Math::Abs( m_value );
        m_value = ( m_value < 0 ) ? delta : -delta;
    }

    // Inverts angle between [0;2Pi] and [-2Pi;0]
    inline Radians GetInverse() const
    {
        Radians r( m_value );
        r.Invert();
        return r;
    }

    // Flips the front and rear 180 degree arc i.e. 135 becomes -45, -90 becomes 90, etc.
    inline void Flip()
    {
        Clamp180();
        float const delta = Math::Pi - Math::Abs( m_value );
        m_value = ( m_value < 0 ) ? delta : -delta;
    }

    // Flips the front and rear 180 degree arc i.e. 135 becomes -45, -90 becomes 90, etc.
    inline Radians GetFlipped() const
    {
        Radians r( m_value );
        r.Flip();
        return r;
    }

private:

    float m_value = 0;
};

inline Degrees::Degrees( Radians const& radians )
    : m_value( radians.ToDegrees() )
{}

inline Radians Degrees::ToRadians() const
{
    return Radians( m_value * Math::DegreesToRadians );
}

inline Degrees::operator Radians() const
{
    return ToRadians();
}

inline bool Degrees::operator>( Radians const& rhs ) const { return m_value > rhs.ToDegrees().m_value; }
inline bool Degrees::operator<( Radians const& rhs ) const { return m_value < rhs.ToDegrees().m_value; }
inline bool Degrees::operator>=( Radians const& rhs ) const { return m_value >= rhs.ToDegrees().m_value; }
inline bool Degrees::operator<=( Radians const& rhs ) const { return m_value <= rhs.ToDegrees().m_value; }

inline bool Degrees::operator==( Radians const& rhs ) const { return Math::IsNearEqual( m_value, rhs.ToDegrees().m_value ); }
inline bool Degrees::operator!=( Radians const& rhs ) const { return !Math::IsNearEqual( m_value, rhs.ToDegrees().m_value ); }

inline Radians::Radians( Degrees const& degrees )
    : m_value( degrees.ToRadians() )
{}

inline bool Radians::operator>( Degrees const& rhs ) const { return m_value > rhs.ToRadians().m_value; }
inline bool Radians::operator<( Degrees const& rhs ) const { return m_value < rhs.ToRadians().m_value; }
inline bool Radians::operator>=( Degrees const& rhs ) const { return m_value >= rhs.ToRadians().m_value; }
inline bool Radians::operator<=( Degrees const& rhs ) const { return m_value <= rhs.ToRadians().m_value; }

inline bool Radians::operator==( Degrees const& rhs ) const { return Math::IsNearEqual( m_value, rhs.ToRadians().m_value ); }
inline bool Radians::operator!=( Degrees const& rhs ) const { return !Math::IsNearEqual( m_value, rhs.ToRadians().m_value ); }

struct EulerAngles
{
public:

    EulerAngles() = default;

    inline explicit EulerAngles( Degrees inX, Degrees inY, Degrees inZ )
        : m_x( inX )
        , m_y( inY )
        , m_z( inZ )
    {}

    inline explicit EulerAngles( Radians inX, Radians inY, Radians inZ )
        : m_x( inX )
        , m_y( inY )
        , m_z( inZ )
    {}

    inline explicit EulerAngles( float inDegreesX, float inDegreesY, float inDegreesZ )
        : m_x( Math::DegreesToRadians * inDegreesX )
        , m_y( Math::DegreesToRadians * inDegreesY )
        , m_z( Math::DegreesToRadians * inDegreesZ )
    {}

    inline EulerAngles( Float3 const& anglesInDegrees )
        : m_x( Math::DegreesToRadians * anglesInDegrees.m_x )
        , m_y( Math::DegreesToRadians * anglesInDegrees.m_y )
        , m_z( Math::DegreesToRadians * anglesInDegrees.m_z )
    {}

    inline void Clamp()
    {
        m_x.Clamp360();
        m_y.Clamp360();
        m_z.Clamp360();
    }

    inline EulerAngles GetClamped() const
    {
        EulerAngles clamped = *this;
        clamped.Clamp();
        return clamped;
    }

    inline Radians GetYaw() const { return m_z; }
    inline Radians GetPitch() const { return m_x; }
    inline Radians GetRoll() const { return m_y; }

    inline Float3 GetAsRadians() const { return Float3( m_x.ToFloat(), m_y.ToFloat(), m_z.ToFloat() ); }
    inline Float3 GetAsDegrees() const { return Float3( m_x.ToDegrees().ToFloat(), m_y.ToDegrees().ToFloat(), m_z.ToDegrees().ToFloat() ); }

    inline bool operator==( EulerAngles const& other ) const { return m_x == other.m_x && m_y == other.m_y && m_z == other.m_z; }
    inline bool operator!=( EulerAngles const& other ) const { return m_x != other.m_x || m_y != other.m_y || m_z != other.m_z; }

    inline Radians& operator[]( uint32_t i ) { return ( (Radians*) this )[i]; }
    inline Radians const& operator[]( uint32_t i ) const { return ( (Radians*) this )[i]; }
    // in degrees
    inline Float3 ToFloat3() const { return Float3( Math::RadiansToDegrees * m_x.ToFloat(), Math::RadiansToDegrees * m_y.ToFloat(), Math::RadiansToDegrees * m_z.ToFloat() ); }

public:

    Radians m_x = 0.0f;
    Radians m_y = 0.0f;
    Radians m_z = 0.0f;
};

struct AxisAngle
{
public:

    inline AxisAngle() = default;
    inline explicit AxisAngle( Float3 axis, Radians angle ) : m_axis( axis ), m_angle( angle ) {}
    inline explicit AxisAngle( Float3 axis, Degrees angle ) : m_axis( axis ), m_angle( angle.ToRadians() ) {}

    inline bool IsValid() const
    {
        float const lengthSq = m_axis.m_x * m_axis.m_x + m_axis.m_y * m_axis.m_y + m_axis.m_z * m_axis.m_z;
        return Math::Abs( lengthSq - 1.0f ) < Math::Epsilon;
    }

public:

    Float3      m_axis = Float3::Zero;
    Radians     m_angle = Radians( 0.0f );
};
