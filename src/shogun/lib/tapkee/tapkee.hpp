/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_MAIN_H_
#define TAPKEE_MAIN_H_

#include <shogun/lib/tapkee/tapkee_defines.hpp>
#include <shogun/lib/tapkee/tapkee_methods.hpp>

namespace tapkee
{

//! Main entry-point of the library. Constructs dense embedding with specified dimension
//! using provided data and callbacks.
//!
//! Has four template parameters:
//! 
//! RandomAccessIterator basic random access iterator with no specific capabilities.
//!
//! KernelCallback that defines DefaultScalarType operator()(RandomAccessIterator, RandomAccessIterator) operation 
//! between two iterators. The operation should return value of Mercer kernel function 
//! between vectors/objects iterators pointing to. KernelCallback should be marked as a kernel function using
//! TAPKEE_CALLBACK_IS_KERNEL macro (fails on compilation in other case).
//!
//! DistanceCallback that defines DefaultScalarType operator()(RandomAccessIterator, RandomAccessIterator) operation
//! between two iterators. DistanceCallback should be marked as a distance function using 
//! TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
//!
//! FeatureVectorCallback that defines void operator()(RandomAccessIterator, DenseVector) operation
//! used to access feature vector pointed by iterator. The callback should put the feature vector pointed by iterator
//! to the vector of second argument.
//!
//! Parameters required by the chosen algorithm are obtained from the parameter map. It fails during runtime if
//! some of required parameters are not specified or have improper values.
//!
//! @param begin begin iterator of data
//! @param end end iterator of data
//! @param kernel_callback the kernel callback described before
//! @param distance_callback the distance callback described before
//! @param feature_vector_callback the feature vector access callback descrbied before 
//! @param options parameter map
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                   KernelCallback kernel_callback, DistanceCallback distance_callback,
                   FeatureVectorCallback feature_vector_callback, ParametersMap options)
{
	ReturnResult return_result;

	TAPKEE_METHOD method;
	try 
	{
		method = options[REDUCTION_METHOD].cast<TAPKEE_METHOD>();
	}
	catch (const anyimpl::bad_any_cast&)
	{
		throw std::runtime_error("Wrong method specified.");
	}

#define CALL_IMPLEMENTATION(X) \
		tapkee_internal::embedding_impl<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,X>().embed(\
		begin,end,kernel_callback,distance_callback,feature_vector_callback,options)
#define HANDLE_IMPLEMENTATION(X) \
	case X: return_result = CALL_IMPLEMENTATION(X); break
#define NO_IMPLEMENTATION_YET printf("Not implemented\n"); exit(EXIT_FAILURE)

	try 
	{
		LoggingSingleton::instance().message_info("Using " + tapkee_internal::get_method_name(method) + " method.");
		switch (method)
		{
			HANDLE_IMPLEMENTATION(KERNEL_LOCALLY_LINEAR_EMBEDDING);
			HANDLE_IMPLEMENTATION(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT);
			HANDLE_IMPLEMENTATION(DIFFUSION_MAP);
			HANDLE_IMPLEMENTATION(MULTIDIMENSIONAL_SCALING);
			HANDLE_IMPLEMENTATION(LANDMARK_MULTIDIMENSIONAL_SCALING);
			HANDLE_IMPLEMENTATION(ISOMAP);
			HANDLE_IMPLEMENTATION(LANDMARK_ISOMAP);
			HANDLE_IMPLEMENTATION(NEIGHBORHOOD_PRESERVING_EMBEDDING);
			HANDLE_IMPLEMENTATION(LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
			HANDLE_IMPLEMENTATION(HESSIAN_LOCALLY_LINEAR_EMBEDDING);
			HANDLE_IMPLEMENTATION(LAPLACIAN_EIGENMAPS);
			HANDLE_IMPLEMENTATION(LOCALITY_PRESERVING_PROJECTIONS);
			HANDLE_IMPLEMENTATION(PCA);
			HANDLE_IMPLEMENTATION(KERNEL_PCA);
			HANDLE_IMPLEMENTATION(STOCHASTIC_PROXIMITY_EMBEDDING);
			HANDLE_IMPLEMENTATION(PASS_THRU);
			HANDLE_IMPLEMENTATION(FACTOR_ANALYSIS);
			default: break;
		}
	}
	catch (const std::bad_alloc& ba)
	{
		LoggingSingleton::instance().message_error("Not enough memory available.");
	}
	catch (const std::exception& ex)
	{
		LoggingSingleton::instance().message_error(string("Some error occured : ") + string(ex.what()));
	}

#undef CALL_IMPLEMENTATION 
#undef HANDLE_IMPLEMENTATION
#undef NO_IMPLEMENTATION_YET

	return return_result;
};

} // namespace tapkee

#endif
