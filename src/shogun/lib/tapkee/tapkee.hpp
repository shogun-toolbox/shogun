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

/** Main entry-point of the library. Constructs dense embedding with specified dimension
* using provided data and callbacks. Returns ReturnType that is essentially a pair of 
* DenseMatrix (embedding of provided data) and ProjectingFunction with 
* corresponding ProjectionImplementation used to project data out of sample.
*
* @tparam RandomAccessIterator basic random access iterator with no specific capabilities.
*
* @tparam KernelCallback that defines 
* @code ScalarType operator()( RandomAccessIterator, RandomAccessIterator) @endcode 
* function of two iterators. This method should return value of Mercer kernel function 
* between vectors/objects iterators pointing to. The callback should be marked as a kernel function using
* @ref tapkee::TAPKEE_CALLBACK_IS_KERNEL macro (fails on compilation in other case).
*
* @tparam DistanceCallback that defines 
* @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
* function of two iterators. The callback should be marked as a distance function using 
* @ref TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
*
* @tparam FeatureVectorCallback that defines 
* @code void operator()(RandomAccessIterator, DenseVector) @endcode function
* used to access feature vector pointed by iterator. The callback should put the feature vector 
* pointed by iterator to the second argument vector.
*
* Parameters required by the chosen algorithm are obtained from the parameter map. It gracefully 
* fails during runtime and throws an exception if some of required 
* parameters are not specified or have improper values.
*
* @param begin begin iterator of data
* @param end end iterator of data
* @param kernel_callback the kernel callback implementing
* @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
*
* Used by the following methods: 
* - @ref tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING
* - @ref tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING
* - @ref tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT
* - @ref tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT
* - @ref tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING
* - @ref tapkee::KERNEL_PCA
*
* @param distance_callback the distance callback implementing
* @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
*
* Used by the following methods: 
* - @ref tapkee::LAPLACIAN_EIGENMAPS
* - @ref tapkee::LOCALITY_PRESERVING_PROJECTIONS
* - @ref tapkee::DIFFUSION_MAP
* - @ref tapkee::ISOMAP
* - @ref tapkee::LANDMARK_ISOMAP
* - @ref tapkee::MULTIDIMENSIONAL_SCALING
* - @ref tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING
* - @ref tapkee::STOCHASTIC_PROXIMITY_EMBEDDING
* - @ref tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING
*
* @param feature_vector_callback the feature vector callback implementing
* @code void operator()(RandomAccessIterator, DenseVector) @endcode function
*
* Used by the following methods:
* - @ref tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING
* - @ref tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT
* - @ref tapkee::LOCALITY_PRESERVING_PROJECTIONS
* - @ref tapkee::PCA
* - @ref tapkee::RANDOM_PROJECTION
* - @ref tapkee::FACTOR_ANALYSIS
* - @ref tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING
* - @ref tapkee::PASS_THRU
*
* @param parameters parameter map containing values with keys from @ref tapkee::TAPKEE_PARAMETERS
*/
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                   KernelCallback kernel_callback, DistanceCallback distance_callback,
                   FeatureVectorCallback feature_vector_callback, ParametersMap parameters)
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	ReturnResult return_result;

	TAPKEE_METHOD method;
	if (!parameters.count(REDUCTION_METHOD))
		throw missed_parameter_error("Dimension reduction wasn't specified");

	try 
	{
		method = parameters[REDUCTION_METHOD].cast<TAPKEE_METHOD>();
	}
	catch (const anyimpl::bad_any_cast&)
	{
		throw wrong_parameter_type_error("Wrong method type specified.");
	}

#define PUT_DEFAULT(KEY,TYPE,VALUE)              \
	if (!parameters.count(KEY))                     \
		parameters[KEY] = static_cast<TYPE>(VALUE); 

	//// defaults
	PUT_DEFAULT(OUTPUT_FEATURE_VECTORS_ARE_COLUMNS,bool,false);
	PUT_DEFAULT(EIGENSHIFT,ScalarType,1e-9);
	PUT_DEFAULT(CHECK_CONNECTIVITY,bool,true);
	//// end of defaults

#undef PUT_DEFAULT

#define CALL_IMPLEMENTATION(X)                                                                                           \
		tapkee_internal::implementation<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,X>()  \
		(begin,end,kernel_callback,distance_callback,feature_vector_callback,parameters)
#define HANDLE_IMPLEMENTATION(X)                          \
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
			HANDLE_IMPLEMENTATION(T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING);
#endif
			case UNKNOWN_METHOD: throw wrong_parameter_error("unknown method"); break;
		}
	}
	catch (const std::bad_alloc&)
	{
		LoggingSingleton::instance().message_error("Not enough memory available.");
		throw not_enough_memory_error("Not enough memory");
	}

#undef CALL_IMPLEMENTATION 
#undef HANDLE_IMPLEMENTATION

	if (parameters[OUTPUT_FEATURE_VECTORS_ARE_COLUMNS].cast<bool>())
		return_result.first.transposeInPlace();

	return return_result;
}

} // End of namespace tapkee

#endif
