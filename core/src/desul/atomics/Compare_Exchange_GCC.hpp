/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_COMPARE_EXCHANGE_GCC_HPP_
#define DESUL_ATOMICS_COMPARE_EXCHANGE_GCC_HPP_
#include "desul/atomics/Common.hpp"

#include <cstdlib>

#ifdef DESUL_HAVE_GCC_ATOMICS
#if !defined(DESUL_HAVE_16BYTE_COMPARE_AND_SWAP) && !defined(__CUDACC__)
// This doesn't work in WSL??
//#define DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
#endif
namespace desul {

namespace Impl {
template<class T>
struct atomic_exchange_available_gcc {
  constexpr static bool value =
#ifndef DESUL_HAVE_LIBATOMIC
    (sizeof(T)==4 ||
#ifdef DESUL_HAVE_16BYTE_COMPARE_AND_SWAP
     sizeof(T)==16 ||
#endif
     sizeof(T)==8 ) &&
#endif
    std::is_trivially_copyable<T>::value;
};
} //namespace Impl

#if defined(__clang__) && (__clang_major__>=7) && !defined(__APPLE__)
// Disable warning for large atomics on clang 7 and up (checked with godbolt)
// error: large atomic operation may incur significant performance penalty [-Werror,-Watomic-alignment]
// https://godbolt.org/z/G7YhqhbG6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Watomic-alignment"
#endif
template<class MemoryOrder, class MemoryScope>
void atomic_thread_fence(MemoryOrder, MemoryScope) {
  __atomic_thread_fence(GCCMemoryOrder<MemoryOrder>::value);
}

template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<Impl::atomic_exchange_available_gcc<T>::value, T>
atomic_exchange(
    T* dest, T value, MemoryOrder, MemoryScope) {
  T return_val;
  __atomic_exchange(
     dest, &value, &return_val, GCCMemoryOrder<MemoryOrder>::value);
  return return_val;
}

//#ifndef __SYCL_DEVICE_ONLY__
// Failure mode for atomic_compare_exchange_n cannot be RELEASE nor ACQREL so
// Those two get handled separatly.
template <typename T, class MemoryOrder, class MemoryScope>
std::enable_if_t<Impl::atomic_exchange_available_gcc<T>::value, T>
atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrder, MemoryScope) {
#ifndef KOKKOS_ENABLE_SYCL
  (void)__atomic_compare_exchange(
      dest, &compare, &value, false, GCCMemoryOrder<MemoryOrder>::value, GCCMemoryOrder<MemoryOrder>::value);
#else
  assert(false);
#endif
  return value;
}

template <typename T, class MemoryScope>
std::enable_if_t<Impl::atomic_exchange_available_gcc<T>::value, T>
atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrderRelease, MemoryScope) {
#ifndef KOKKOS_ENABLE_SYCL
  (void)__atomic_compare_exchange(
      dest, &compare, &value, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
#else
  assert(false);
#endif
  return value;
}

template <typename T, class MemoryScope>
std::enable_if_t<Impl::atomic_exchange_available_gcc<T>::value, T>
atomic_compare_exchange(
    T* dest, T compare, T value, MemoryOrderAcqRel, MemoryScope) {
#ifndef KOKKOS_ENABLE_SYCL
  (void)__atomic_compare_exchange(
      dest, &compare, &value, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#else
  assert(false);
#endif
  return value;
}
//#endif

#if defined(__clang__) && (__clang_major__>=7) && !defined(__APPLE__)
#pragma GCC diagnostic pop
#endif
}  // namespace desul
#endif
#endif
