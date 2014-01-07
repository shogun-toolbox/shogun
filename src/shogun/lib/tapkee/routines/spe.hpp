/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando J. Iglesias Garcia
 */

#ifndef TAPKEE_SPE_H_
#define TAPKEE_SPE_H_

/* Tapkee includes */
#include <lib/tapkee/defines.hpp>
#include <lib/tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator, class PairwiseCallback>
DenseMatrix spe_embedding(RandomAccessIterator begin, RandomAccessIterator end,
		PairwiseCallback callback, const Neighbors& neighbors,
		IndexType target_dimension, bool global_strategy,
		ScalarType tolerance, int nupdates, IndexType max_iter)
{
	timed_context context("SPE embedding computation");
	IndexType k = 0;
	if (!global_strategy)
		k = neighbors[0].size();

	// The number of data points
	int N = end-begin;
	while (nupdates > N/2)
		nupdates = N/2;

	// Look for the maximum distance
	ScalarType max = 0.0;
	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter+1; j_iter!=end; ++j_iter)
		{
			max = std::max(max, callback.distance(*i_iter,*j_iter));
		}
	}

	// Distances normalizer used in global strategy
	ScalarType alpha = 0.0;
	if (global_strategy)
		alpha = 1.0 / max * std::sqrt(2.0);

	// Random embedding initialization, Y is the short for embedding_feature_matrix
	DenseMatrix Y = (DenseMatrix::Random(target_dimension,N)
		       + DenseMatrix::Ones(target_dimension,N)) / 2;
	// Auxiliary diffference embedding feature matrix
	DenseMatrix Yd(target_dimension,nupdates);

	// SPE's main loop

	typedef std::vector<int> Indices;
	typedef std::vector<int>::iterator IndexIterator;

	// Maximum number of iterations
	if (max_iter == 0)
	{
		max_iter = 2000 + static_cast<IndexType>(floor(0.04 * N*N));
		if (!global_strategy)
			max_iter *= 3;
	}

	// Learning parameter
	ScalarType lambda = 1.0;
	// Vector of indices used for shuffling
	Indices indices(N);
	for (int i=0; i<N; ++i)
		indices[i] = i;
	// Vector with distances in the original space of the points to update
	DenseVector Rt(nupdates);
	DenseVector scale(nupdates);
	DenseVector D(nupdates);
	// Pointers to the indices of the elements to update
	IndexIterator ind1;
	IndexIterator ind2;
	// Helper used in local strategy
	Indices ind1Neighbors;
	if (!global_strategy)
		ind1Neighbors.resize(k*nupdates);

	for (IndexType i=0; i<max_iter; ++i)
	{
		// Shuffle to select the vectors to update in this iteration
		tapkee::random_shuffle(indices.begin(),indices.end());

		ind1 = indices.begin();
		ind2 = indices.begin()+nupdates;

		// With local strategy, the seecond set of indices is selected among
		// neighbors of the first set
		if (!global_strategy)
		{
			// Neighbors of interest
			for(int j=0; j<nupdates; ++j)
			{
				const LocalNeighbors& current_neighbors =
					neighbors[*ind1++];

				for(IndexType kk=0; kk<k; ++kk)
					ind1Neighbors[kk + j*k] = current_neighbors[kk];
			}
			// Restore ind1
			ind1 = indices.begin();

			// Generate pseudo-random indices and select final indices
			for(int j=0; j<nupdates; ++j)
			{
				IndexType r = static_cast<IndexType>(floor(tapkee::uniform_random()*(k-1)) + k*j);
				indices[nupdates+j] = ind1Neighbors[r];
			}
		}


		// Compute distances between the selected points in the embedded space
		for(int j=0; j<nupdates; ++j)
		{
			//FIXME it seems that here Euclidean distance is forced
			D[j] = (Y.col(*ind1) - Y.col(*ind2)).norm();
			++ind1, ++ind2;
		}

		// Get the corresponding distances in the original space
		if (global_strategy)
			Rt.fill(alpha);
		else // local_strategy
			Rt.fill(1);

		ind1 = indices.begin();
		ind2 = indices.begin()+nupdates;
		for (int j=0; j<nupdates; ++j)
			Rt[j] *= callback.distance(*(begin + *ind1++), *(begin + *ind2++));

		// Compute some terms for update

		// Scale factor
		D.array() += tolerance;
		scale = (Rt-D).cwiseQuotient(D);

		ind1 = indices.begin();
		ind2 = indices.begin()+nupdates;
		// Difference matrix
		for (int j=0; j<nupdates; ++j)
		{
			Yd.col(j).noalias() = Y.col(*ind1) - Y.col(*ind2);

			++ind1, ++ind2;
		}

		ind1 = indices.begin();
		ind2 = indices.begin()+nupdates;
		// Update the location of the vectors in the embedded space
		for (int j=0; j<nupdates; ++j)
		{
			Y.col(*ind1) += lambda / 2 * scale[j] * Yd.col(j);
			Y.col(*ind2) -= lambda / 2 * scale[j] * Yd.col(j);

			++ind1, ++ind2;
		}

		// Update the learning parameter
		lambda = lambda - ( lambda / max_iter );
	}

	return Y.transpose();
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif /* TAPKEE_SPE_H_ */
