/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_MAIN_H_
#define TAPKEE_MAIN_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/tapkee_defines.hpp>
#include <shogun/lib/tapkee/tapkee_methods.hpp>
/* End of Tapkee includes */

namespace tapkee
{

//! Main entry-point of the library. Constructs dense embedding with specified dimension
//! using provided data and callbacks. Returns ReturnType that is essentially a pair of 
//! DenseMatrix (embedding of provided data) and ProjectingFunction with 
//! corresponding ProjectionImplementation used to project data out of sample.
//!
//! @tparam RandomAccessIterator basic random access iterator with no specific capabilities.
//!
//! @tparam KernelCallback that defines 
//! @code ScalarType operator()( RandomAccessIterator, RandomAccessIterator) @endcode 
//! function of two iterators. This method should return value of Mercer kernel function 
//! between vectors/objects iterators pointing to. The callback should be marked as a kernel function using
//! @ref tapkee::TAPKEE_CALLBACK_IS_KERNEL macro (fails on compilation in other case).
//!
//! @tparam DistanceCallback that defines 
//! @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
//! function of two iterators. The callback should be marked as a distance function using 
//! @ref TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
//!
//! @tparam FeatureVectorCallback that defines 
//! @code void operator()(RandomAccessIterator, DenseVector) @endcode function
//! used to access feature vector pointed by iterator. The callback should put the feature vector 
//! pointed by iterator to the second argument vector.
//!
//! Parameters required by the chosen algorithm are obtained from the parameter map. It gracefully 
//! fails during runtime and throws an exception if some of required 
//! parameters are not specified or have improper values.
//!
//! @param begin begin iterator of data
//! @param end end iterator of data
//! @param kernel_callback the kernel callback described before
//! @param distance_callback the distance callback described before
//! @param feature_vector_callback the feature vector access callback descrbied before 
//! @param options parameter map
//!
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                   KernelCallback kernel_callback, DistanceCallback distance_callback,
                   FeatureVectorCallback feature_vector_callback, ParametersMap options)
{
	ReturnResult return_result;

	TAPKEE_METHOD method;
	if (!options.count(REDUCTION_METHOD))
		throw missed_parameter_error("Dimension reduction wasn't specified");

	try 
	{
		method = options[REDUCTION_METHOD].cast<TAPKEE_METHOD>();
	}
	catch (const anyimpl::bad_any_cast&)
	{
		throw wrong_parameter_type_error("Wrong method type specified.");
	}

#define CALL_IMPLEMENTATION(X) \
		tapkee_internal::implementation<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,X>()(\
		begin,end,kernel_callback,distance_callback,feature_vector_callback,options)
#define HANDLE_IMPLEMENTATION(X) \
	case X: return_result = CALL_IMPLEMENTATION(X); break

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
			HANDLE_IMPLEMENTATION(RANDOM_PROJECTION);
			HANDLE_IMPLEMENTATION(STOCHASTIC_PROXIMITY_EMBEDDING);
			HANDLE_IMPLEMENTATION(PASS_THRU);
			HANDLE_IMPLEMENTATION(FACTOR_ANALYSIS);
#ifdef TAPKEE_USE_GPL_TSNE
			HANDLE_IMPLEMENTATION(TSNE);
#endif
			case UNKNOWN_METHOD: throw wrong_parameter_error("unknown method"); break;
		}
	}
	catch (const std::bad_alloc& ba)
	{
		LoggingSingleton::instance().message_error("Not enough memory available.");
		throw not_enough_memory_error("Not enough memory");
	}

#undef CALL_IMPLEMENTATION 
#undef HANDLE_IMPLEMENTATION

	return return_result;
};

} // namespace tapkee

#endif
