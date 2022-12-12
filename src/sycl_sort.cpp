#ifdef SYCL_SORT
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <omp.h>
#include <map>
#include <iostream>

//namespace sycl = cl::sycl;

// This file was written with the help of Thomas Applencourt, and follows his guide from:
// https://github.com/argonne-lcf/HPC-Patterns/blob/main/sycl_omp_ze_interopt/interop_omp_sycl.cpp

sycl::queue get_sycl_queue()
{
  omp_interop_t o = 0;
  int D = omp_get_default_device();
  #pragma omp interop init(prefer_type("sycl"), targetsync: o) device(D)
  int err = -1;
  auto* sycl_context = static_cast<sycl::context *>(omp_get_interop_ptr(o, omp_ipr_device_context, &err));
  assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device_context)");
  auto* sycl_device =  static_cast<sycl::device *>(omp_get_interop_ptr(o, omp_ipr_device, &err));
  assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device)");
  return sycl::queue(*sycl_context, *sycl_device);
}

#include "openmc/event.h"
namespace openmc{

void sort_queue_SYCL(EventQueueItem* begin, EventQueueItem* end)
{
  static sycl::queue Q = get_sycl_queue();
  //std::sort( oneapi::dpl::execution::make_device_policy(Q), begin, end);
  std::sort( oneapi::dpl::execution::dpcpp_default, begin, end);
}

} // end namespace openmc

#endif
