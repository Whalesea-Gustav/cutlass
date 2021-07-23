/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Defines a proxy class for storing Tensor Float 32 data type.
*/
#pragma once

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <limits>
#include <cstdint>
#endif

#include "cutlass/cutlass.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Tensor Float 32 data type
struct alignas(4) halfhalf_t {

  //
  // Data members
  //

  /// Storage type
  uint32_t storage;

  //
  // Methods
  //

  /// Constructs from an unsigned int
  CUTLASS_HOST_DEVICE
  static halfhalf_t bitcast(uint32_t x) {
    halfhalf_t h;
    h.storage = x;
    return h;
  }

  /// Emulated rounding is fast in device code
  CUTLASS_HOST_DEVICE
  static halfhalf_t round_half_ulp_truncate(float const &s) {
	const auto hv = __float2half(s);
	const auto dhv = __float2half((s - __half2float(hv)) * 2048.f);

	const half2 h2 = {hv, dhv};
	const auto x = *reinterpret_cast<const uint32_t*>(&h2);

    return halfhalf_t::bitcast(x);
  }

  /// Default constructor
  CUTLASS_HOST_DEVICE
  halfhalf_t() : storage(0) { }

  /// Floating-point conversion - round toward nearest even
  CUTLASS_HOST_DEVICE
  explicit halfhalf_t(float x): storage(round_half_ulp_truncate(x).storage) { }

  /// Floating-point conversion - round toward nearest even
  CUTLASS_HOST_DEVICE
  explicit halfhalf_t(double x): halfhalf_t(float(x)) {

  }

  /// Integer conversion - round toward zero
  CUTLASS_HOST_DEVICE
  explicit halfhalf_t(int x) {
    float flt = static_cast<float>(x);
    storage = reinterpret_cast<uint32_t const &>(flt);
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator float() const {
	const auto x = *reinterpret_cast<const half2*>(&storage);
	const auto hv = __high2float(x);
	const auto dhv = __low2float(x);
    
	return hv + dhv / 2048.f;
  }

  /// Converts to float
  CUTLASS_HOST_DEVICE
  operator double() const {
    return double(float(*this));
  }

  /// Converts to int
  CUTLASS_HOST_DEVICE
  explicit operator int() const {
    return int(float(*this));
  }

  /// Casts to bool
  CUTLASS_HOST_DEVICE
  operator bool() const {
    return (float(*this) != 0.0f);
  }

  /// Obtains raw bits
  CUTLASS_HOST_DEVICE
  uint32_t raw() const {
    return storage;
  }

  /// Returns the sign bit
  CUTLASS_HOST_DEVICE
  bool signbit() const {
    return ((raw() & 0x80000000) != 0);
  }

  /// Returns the biased exponent
  CUTLASS_HOST_DEVICE
  int exponent_biased() const {
    return 0;
  }

  /// Returns the unbiased exponent
  CUTLASS_HOST_DEVICE
  int exponent() const {
    return 0;
  }

  /// Returns the mantissa
  CUTLASS_HOST_DEVICE
  int mantissa() const {
    return 0;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool signbit(cutlass::halfhalf_t const& h) {
  return h.signbit();
}

CUTLASS_HOST_DEVICE
cutlass::halfhalf_t abs(cutlass::halfhalf_t const& h) {
  return cutlass::halfhalf_t::bitcast(h.raw() & 0x7fffffff);
}

CUTLASS_HOST_DEVICE
bool isnan(cutlass::halfhalf_t const& h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isfinite(cutlass::halfhalf_t const& h) {
  return (h.exponent_biased() != 0x0ff);
}

CUTLASS_HOST_DEVICE
cutlass::halfhalf_t nan_halfhalf(const char*) {
  // NVIDIA canonical NaN
  return cutlass::halfhalf_t::bitcast(0x7fffffff);
}

CUTLASS_HOST_DEVICE
bool isinf(cutlass::halfhalf_t const& h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

CUTLASS_HOST_DEVICE
bool isnormal(cutlass::halfhalf_t const& h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

CUTLASS_HOST_DEVICE
int fpclassify(cutlass::halfhalf_t const& h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x0ff) {
    if (mantissa) {
      return FP_NAN;
    }
    else {
      return FP_INFINITE;
    }
  }
  else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    }
    else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

CUTLASS_HOST_DEVICE
cutlass::halfhalf_t sqrt(cutlass::halfhalf_t const& h) {
#if defined(__CUDACC_RTC__)
  return cutlass::halfhalf_t(sqrtf(float(h)));
#else
  return cutlass::halfhalf_t(std::sqrt(float(h)));
#endif
}

CUTLASS_HOST_DEVICE
halfhalf_t copysign(halfhalf_t const& a, halfhalf_t const& b) {

  uint32_t a_mag = (reinterpret_cast<uint32_t const &>(a) & 0x7fffffff);  
  uint32_t b_sign = (reinterpret_cast<uint32_t const &>(b) & 0x80000000);
  uint32_t result = (a_mag | b_sign);

  return reinterpret_cast<halfhalf_t const &>(result);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {

#if !defined(__CUDACC_RTC__)
/// Numeric limits
template <>
struct numeric_limits<cutlass::halfhalf_t> {
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_infinity = true;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = 19;

  /// Least positive value
  static cutlass::halfhalf_t min() { return cutlass::halfhalf_t::bitcast(0x01); }

  /// Minimum finite value
  static cutlass::halfhalf_t lowest() { return cutlass::halfhalf_t::bitcast(0xff7fffff); }

  /// Maximum finite value
  static cutlass::halfhalf_t max() { return cutlass::halfhalf_t::bitcast(0x7f7fffff); }

  /// Returns smallest finite value
  static cutlass::halfhalf_t epsilon() { return cutlass::halfhalf_t::bitcast(0x1000); }

  /// Returns smallest finite value
  static cutlass::halfhalf_t round_error() { return cutlass::halfhalf_t(0.5f); }

  /// Returns smallest finite value
  static cutlass::halfhalf_t infinity() { return cutlass::halfhalf_t::bitcast(0x7f800000); }

  /// Returns smallest finite value
  static cutlass::halfhalf_t quiet_NaN() { return cutlass::halfhalf_t::bitcast(0x7fffffff); }

  /// Returns smallest finite value
  static cutlass::halfhalf_t signaling_NaN() { return cutlass::halfhalf_t::bitcast(0x7fffffff); }

  /// Returns smallest finite value
  static cutlass::halfhalf_t denorm_min() { return cutlass::halfhalf_t::bitcast(0x1); }
};
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace std

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
bool operator==(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return float(lhs) == float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator!=(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return float(lhs) != float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return float(lhs) < float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator<=(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return float(lhs) <= float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return float(lhs) > float(rhs);
}

CUTLASS_HOST_DEVICE
bool operator>=(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return float(lhs) >= float(rhs);
}

CUTLASS_HOST_DEVICE
halfhalf_t operator+(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return halfhalf_t(float(lhs) + float(rhs));
}


CUTLASS_HOST_DEVICE
halfhalf_t operator-(halfhalf_t const& lhs) {
  float x = -reinterpret_cast<float const &>(lhs);
  return reinterpret_cast<halfhalf_t const &>(x);
}

CUTLASS_HOST_DEVICE
halfhalf_t operator-(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return halfhalf_t(float(lhs) - float(rhs));
}

CUTLASS_HOST_DEVICE
halfhalf_t operator*(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return halfhalf_t(float(lhs) * float(rhs));
}

CUTLASS_HOST_DEVICE
halfhalf_t operator/(halfhalf_t const& lhs, halfhalf_t const& rhs) {
  return halfhalf_t(float(lhs) / float(rhs));
}

CUTLASS_HOST_DEVICE
halfhalf_t& operator+=(halfhalf_t & lhs, halfhalf_t const& rhs) {
  lhs = halfhalf_t(float(lhs) + float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
halfhalf_t& operator-=(halfhalf_t & lhs, halfhalf_t const& rhs) {
  lhs = halfhalf_t(float(lhs) - float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
halfhalf_t& operator*=(halfhalf_t & lhs, halfhalf_t const& rhs) {
  lhs = halfhalf_t(float(lhs) * float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
halfhalf_t& operator/=(halfhalf_t & lhs, halfhalf_t const& rhs) {
  lhs = halfhalf_t(float(lhs) / float(rhs));
  return lhs;
}

CUTLASS_HOST_DEVICE
halfhalf_t& operator++(halfhalf_t & lhs) {
  float tmp(lhs);
  ++tmp;
  lhs = halfhalf_t(tmp);
  return lhs;
}

CUTLASS_HOST_DEVICE
halfhalf_t& operator--(halfhalf_t & lhs) {
  float tmp(lhs);
  --tmp;
  lhs = halfhalf_t(tmp);
  return lhs;
}

CUTLASS_HOST_DEVICE
halfhalf_t operator++(halfhalf_t & lhs, int) {
  halfhalf_t ret(lhs);
  float tmp(lhs);
  tmp++;
  lhs = halfhalf_t(tmp);
  return ret;
}

CUTLASS_HOST_DEVICE
halfhalf_t operator--(halfhalf_t & lhs, int) {
  halfhalf_t ret(lhs);
  float tmp(lhs);
  tmp--;
  lhs = halfhalf_t(tmp);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

CUTLASS_HOST_DEVICE
cutlass::halfhalf_t operator "" _halfhalf(long double x) {
  return cutlass::halfhalf_t(float(x));
}

CUTLASS_HOST_DEVICE
cutlass::halfhalf_t operator "" _halfhalf(unsigned long long int x) {
  return cutlass::halfhalf_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
