/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_SYCL_INTERNAL_HPP
#define KOKKOS_SYCL_INTERNAL_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_SYCL

#include <Cuda/Kokkos_Cuda_Error.hpp>

namespace Kokkos {
namespace Impl {

// Note: hardcoded for Compute 8.0
// https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
inline int sycl_max_active_blocks_per_sm(int block_size, size_t dynamic_shmem, const int regs_per_thread) {
  // Limits due do registers/SM
  // int const regs_per_sm = properties.regsPerMultiprocessor;
  int const regs_per_sm     = 65536;
  int const max_blocks_regs = regs_per_sm / (regs_per_thread * block_size);

  // Limits due to shared memory/SM
  // size_t const shmem_per_sm            =
  // properties.sharedMemPerMultiprocessor; size_t const shmem_per_block =
  // properties.sharedMemPerBlock;
  size_t const shmem_per_sm            = 167936;
  size_t const shmem_per_block         = 167936;

  // size_t const static_shmem            = attributes.sharedSizeBytes;
  // size_t const dynamic_shmem_per_block = attributes.maxDynamicSharedSizeBytes;
  size_t const static_shmem            = 0;
  size_t const dynamic_shmem_per_block = 0;


  size_t const total_shmem             = static_shmem + dynamic_shmem;

  int const max_blocks_shmem =
      total_shmem > shmem_per_block || dynamic_shmem > dynamic_shmem_per_block
          ? 0
          : (total_shmem > 0 ? (int)shmem_per_sm / total_shmem
                             : max_blocks_regs);

  // Overall occupancy in blocks
  return std::min({max_blocks_regs, max_blocks_shmem, max_blocks_per_sm});
}


template <typename UnaryFunction, typename FunctorType, typename LaunchBounds, template <typename> class Wrapper>
inline int sycl_deduce_block_size(bool early_termination,
				  const sycl::queue& q,
				  const FunctorType& f,
                                  UnaryFunction block_size_to_dynamic_shmem,
                                  LaunchBounds) {

  // TODO:
  // TODO maxThreadsPerMultiProcessor
  // TODO max_active_blocks_per_sm

  // Get the device & compiled kernel
  const sycl::device sycl_device = q.get_device();
  sycl::program p{q.get_context()};
  p.build_with_kernel_type<Wrapper<FunctorType>>();
  auto k = p.get_kernel<Wrapper<FunctorType>>();

  // Get device-specific kernel info (number of registers & max work group size)
  auto num_regs = k.template get_info<
      sycl::info::kernel_device_specific::ext_codeplay_num_regs>(sycl_device);

  size_t kernelMaxThreadsPerBlock = k.template get_info<
      sycl::info::kernel_device_specific::work_group_size>(sycl_device);

  // Limits
  int const max_threads_per_sm = 2048; //dQ

  int const device_max_threads_per_block =
    sycl_device.template get_info<sycl::info::device::max_work_group_size>();

  int const max_threads_per_block =
      std::min(LaunchBounds::maxTperB == 0 ? (int)device_max_threads_per_block
                                           : (int)LaunchBounds::maxTperB,
               (int)kernelMaxThreadsPerBlock);

  int const min_blocks_per_sm =
      LaunchBounds::minBperSM == 0 ? 1 : LaunchBounds::minBperSM;

  // Recorded maximum
  int opt_block_size     = 0;
  int opt_threads_per_sm = 0;

  // TODO generalise 32?
  // info::kernel_device_specific::preferred_work_group_size_multiple
  for (int block_size = max_threads_per_block; block_size > 0;
       block_size -= 32) {

    // 'dynamic_shmem' is a misnomer here. It's allocated before launch by
    // the host & it's sycl 'local' memory.
    size_t const dynamic_shmem = block_size_to_dynamic_shmem(block_size);

    // TODO max_active_blocks_per_sm
    int blocks_per_sm = sycl_max_active_blocks_per_sm(
        block_size, dynamic_shmem, num_regs);

    int threads_per_sm = blocks_per_sm * block_size;
    if (threads_per_sm > max_threads_per_sm) {
      blocks_per_sm  = max_threads_per_sm / block_size;
      threads_per_sm = blocks_per_sm * block_size;
    }

    if (blocks_per_sm >= min_blocks_per_sm) {
      if (threads_per_sm >= opt_threads_per_sm) {
        opt_block_size     = block_size;
        opt_threads_per_sm = threads_per_sm;
      }
    }

    if (early_termination && opt_block_size != 0) break;
  }

  return opt_block_size;
}

template <class FunctorType, class LaunchBounds,
          template <typename> class Wrapper>
int sycl_get_opt_block_size(const sycl::queue& q, const FunctorType& f,
                            const size_t vector_length,
                            const size_t shmem_block,
                            const size_t shmem_thread) {

  // TODO - cuda equiv here calls:
  // auto const& prop = Kokkos::Cuda().cuda_device_prop();
  // i.e. device info caching.

  auto const block_size_to_dynamic_shmem = [&f, vector_length, shmem_block,
                                            shmem_thread](int block_size) {
    size_t const functor_shmem =
        Kokkos::Impl::FunctorTeamShmemSize<FunctorType>::value(
            f, block_size / vector_length);

    size_t const dynamic_shmem = shmem_block +
                                 shmem_thread * (block_size / vector_length) +
                                 functor_shmem;
    return dynamic_shmem;
  };


  return sycl_deduce_block_size<decltype(block_size_to_dynamic_shmem), FunctorType, LaunchBounds, Wrapper>(false, q, f, block_size_to_dynamic_shmem, LaunchBounds{});
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_SYCL
#endif  /* #ifndef KOKKOS_SYCL_INTERNAL_HPP */
