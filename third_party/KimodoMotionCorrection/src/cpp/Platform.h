/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Finds the current platform
#if defined( __WIN32__ ) || defined( _WIN32 )
#    define PLATFORM_WIN32
#else
#    define PLATFORM_LINUX
#endif

//
// Platform Specific Helpers/Functions
//

// DLL export
#if defined(PLATFORM_WIN32) // Windows
#    if defined(COMPILER_MSVC)
#        if defined(STATIC_LIB)
#            define API
#        else
#            if defined(API)
#                define API __declspec(dllexport)
#            else
#                define API __declspec(dllimport)
#            endif
#        endif
#    else
#        if defined(STATIC_LIB)
#            define API
#        else
#            if defined(API)
#                define API __attribute__ ((dllexport))
#            else
#                define API __attribute__ ((dllimport))
#            endif
#        endif
#    endif
#    define DISABLE_OPTIMIZATION __pragma( optimize( "", off ) )
#    define ENABLE_OPTIMIZATION __pragma( optimize( "", on ) )
#    define DEBUG_BREAK() // __debugbreak()
#else // Linux settings
#    include <signal.h>
#    define API __attribute__ ((visibility ("default")))
#    define DISABLE_OPTIMIZATION
#    define ENABLE_OPTIMIZATION
#    define DEBUG_BREAK() // raise(SIGTRAP)
#endif
