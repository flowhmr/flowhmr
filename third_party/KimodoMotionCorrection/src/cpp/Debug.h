/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Platform.h"

#define ASSERT( cond ) do { if( !(cond) ) { DEBUG_BREAK(); } } while( 0 )
#define HALT() { DEBUG_BREAK(); }
#define UNIMPLEMENTED_FUNCTION() { DEBUG_BREAK(); }
#define UNREACHABLE_CODE() { DEBUG_BREAK(); }
