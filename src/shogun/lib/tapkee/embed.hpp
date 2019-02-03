/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_EMBED_H_
#define TAPKEE_EMBED_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
#include <shogun/lib/tapkee/methods.hpp>
/* End of Tapkee includes */

namespace tapkee
{
/** Constructs a dense embedding with specified
 * dimensionality using provided data represented by random access iterators
 * and provided callbacks. Returns ReturnType that is essentially a pair of
 * @ref DenseMatrix (embedding of provided data) and a ProjectingFunction with
 * corresponding ProjectionImplementation used to project
 * data out of the sample.
 *
 * @tparam RandomAccessIterator random access iterator with no
 *         specific capabilities that points to some RandomAccessIterator::value_type
 *         (the simplest case is RandomAccessIterator::value_type being int).
 *
 * @tparam KernelCallback a callback that defines
 * @code ScalarType kernel(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
 * function of two iterators. This method should return value of Mercer kernel function
 * between vectors/objects iterators pointing to.
 *
 * @tparam DistanceCallback a callback that defines
 * @code ScalarType distance(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
 * function of two iterators.
 *
 * @tparam FeaturesCallback a callback that defines
 * @code void vector(const RandomAccessIterator::value_type&, DenseVector&) @endcode function
 * used to access feature vector pointed by iterator. The callback should put the feature vector
 * pointed by the iterator to the provided vector.
 *
 * Parameters required by the chosen algorithm are obtained from the parameter map. It gracefully
 * fails during runtime and throws an exception if some of required
 * parameters are not specified or have improper values.
 *
 * @param begin begin iterator of data
 * @param end end iterator of data
 * @param kernel_callback the kernel callback implementing
 * @code ScalarType kernel(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
 * Used by the following methods:
 * - @ref tapkee::KernelLocallyLinearEmbedding
 * - @ref tapkee::NeighborhoodPreservingEmbedding
 * - @ref tapkee::KernelLocalTangentSpaceAlignment
 * - @ref tapkee::LinearLocalTangentSpaceAlignment
 * - @ref tapkee::HessianLocallyLinearEmbedding
 * - @ref tapkee::KernelPCA
 *
 * @param distance_callback the distance callback implementing
 * @code ScalarType distance(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
 * Used by the following methods:
 * - @ref tapkee::LaplacianEigenmaps
 * - @ref tapkee::LocalityPreservingProjections
 * - @ref tapkee::DiffusionMap
 * - @ref tapkee::Isomap
 * - @ref tapkee::LandmarkIsomap
 * - @ref tapkee::MultidimensionalScaling
 * - @ref tapkee::LandmarkMultidimensionalScaling
 * - @ref tapkee::StochasticProximityEmbedding
 * - @ref tapkee::tDistributedStochasticNeighborEmbedding
 *
 * @param feature_vector_callback the feature vector callback implementing
 * @code void vector(const RandomAccessIterator::value_type&, DenseVector&) @endcode
 * Used by the following methods:
 * - @ref tapkee::NeighborhoodPreservingEmbedding
 * - @ref tapkee::LinearLocalTangentSpaceAlignment
 * - @ref tapkee::LocalityPreservingProjections
 * - @ref tapkee::PCA
 * - @ref tapkee::RandomProjection
 * - @ref tapkee::FactorAnalysis
 * - @ref tapkee::tDistributedStochasticNeighborEmbedding
 * - @ref tapkee::PassThru
 *
 * @param parameters a set of parameters formed with
 *        keywords expression.
 *
 * @throw tapkee::wrong_parameter_error if wrong parameter value is passed
 * @throw tapkee::missed_parameter_error if some required parameter is missed
 * @throw tapkee::multiple_parameter_error if some parameter is provided more than once
 * @throw tapkee::unsupported_method_error if some method or combination of methods is unsupported
 * @throw tapkee::not_enough_memory_error if there is not enough memory to perform the computations
 * @throw tapkee::cancelled_exception if computations were cancelled due to cancel_function returned true
 * @throw tapkee::eigendecomposition_error if eigendecomposition has failed
 *
 */
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>
TapkeeOutput embed(RandomAccessIterator begin, RandomAccessIterator end,
                   KernelCallback kernel_callback, DistanceCallback distance_callback,
                   FeaturesCallback features_callback, stichwort::ParametersSet parameters)
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	TapkeeOutput output;

	try
	{
		parameters.check();
		parameters.merge(tapkee_internal::defaults);

		DimensionReductionMethod selected_method = parameters[method];

		void (*progress_function_ptr)(double) = parameters[progress_function];
		bool (*cancel_function_ptr)() = parameters[cancel_function];

		tapkee_internal::Context context(progress_function_ptr,cancel_function_ptr);

		LoggingSingleton::instance().message_info(formatting::format("Using the {} method.", get_method_name(selected_method)));

		output = tapkee_internal::initialize(begin,end,kernel_callback,distance_callback,features_callback,parameters,context)
								 .embedUsing(selected_method);
	}
	catch (const std::bad_alloc&)
	{
		throw tapkee::not_enough_memory_error("Not enough memory");
	}
	catch (const stichwort::wrong_parameter_error& ex)
	{
		throw tapkee::wrong_parameter_error(ex.what());
	}
	catch (const stichwort::wrong_parameter_type_error& ex)
	{
		throw tapkee::wrong_parameter_type_error(ex.what());
	}
	catch (const stichwort::multiple_parameter_error& ex)
	{
		throw tapkee::multiple_parameter_error(ex.what());
	}
	catch (const stichwort::missed_parameter_error& ex)
	{
		throw tapkee::missed_parameter_error(ex.what());
	}

	return output;
}
}
#endif
