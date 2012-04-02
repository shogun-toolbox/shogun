/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/converter/StochasticProximityEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distance/Distance.h>

using namespace shogun;

class SPE_COVERTREE_POINT
{
public:

	SPE_COVERTREE_POINT(int32_t index, const SGMatrix<float64_t>& dmatrix)
	{
		point_index = index;
		distance_matrix = dmatrix;
	}

	inline double distance(const SPE_COVERTREE_POINT& p) const
	{
		return distance_matrix[point_index*distance_matrix.num_rows+p.point_index];
	}

	inline bool operator==(const SPE_COVERTREE_POINT& p) const
	{
		return (p.point_index==point_index);
	}

	int32_t point_index;
	SGMatrix<float64_t> distance_matrix;
};

CStochasticProximityEmbedding::CStochasticProximityEmbedding() : 
	CEmbeddingConverter()
{
	// Initialize to default values
	m_k         = 12;
	m_nupdates  = 100;
	m_strategy  = SPE_GLOBAL;
	m_tolerance = 1e-5;

	init();
}

void CStochasticProximityEmbedding::init()
{
	SG_ADD(&m_k, "m_k", "Number of neighbors", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_strategy, "m_strategy", "SPE strategy", 
			MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "m_tolerance", "Regularization parameter", 
			MS_NOT_AVAILABLE);
}

CStochasticProximityEmbedding::~CStochasticProximityEmbedding()
{
}

void CStochasticProximityEmbedding::set_k(int32_t k)
{
	if ( k <= 0 )
		SG_ERROR("Number of neighbors k must be greater than 0");

	m_k = k;
}

int32_t CStochasticProximityEmbedding::get_k() const
{
	return m_k;
}

void CStochasticProximityEmbedding::set_strategy(ESPEStrategy strategy)
{
	m_strategy = strategy;
}

ESPEStrategy CStochasticProximityEmbedding::get_strategy() const
{
	return m_strategy;
}

void CStochasticProximityEmbedding::set_tolerance(float32_t tolerance)
{
	if ( tolerance <= 0 )
		SG_ERROR("Tolerance regularization parameter must be greater "
			 "than 0");

	m_tolerance = tolerance;
}

int32_t CStochasticProximityEmbedding::get_tolerance() const
{
	return m_tolerance;
}

void CStochasticProximityEmbedding::set_nupdates(int32_t nupdates)
{
	if ( nupdates <= 0 )
		SG_ERROR("The number of updates must be greater than 0");

	m_nupdates = nupdates;
}

int32_t CStochasticProximityEmbedding::get_nupdates() const
{
	return m_nupdates;
}

const char * CStochasticProximityEmbedding::get_name() const
{
	return "StochasticProximityEmbedding";
}

CFeatures* CStochasticProximityEmbedding::apply(CFeatures* features)
{
	if ( !features )
		SG_ERROR("Features are required to apply SPE\n");

	// Shorthand for the SimpleFeatures
	CSimpleFeatures< float64_t >* simple_features = 
		(CSimpleFeatures< float64_t >*) features;
	SG_REF(features);

	// Get and check the number of vectors
	int32_t N = simple_features->get_num_vectors();
	if ( m_strategy == SPE_LOCAL && m_k >= N )
		SG_ERROR("The number of neighbors (%d) must be less than "
		         "the number of vectors (%d)\n", m_k, N);

	if ( 2*m_nupdates > N )
		SG_ERROR("The number of vectors (%d) must be at least two times "
			 "the number of updates (%d)\n", N, m_nupdates);

	// Compute distance matrix
	SG_DEBUG("Computing distance matrix\n");

	if ( m_distance->get_distance_type() != D_EUCLIDIAN )
		SG_ERROR("SPE only supports Euclidian distance, %s given\n", 
				m_distance->get_name());

	m_distance->init(simple_features, simple_features);
	SGMatrix< float64_t > distance_matrix = m_distance->get_distance_matrix();
	m_distance->remove_lhs_and_rhs();

	// Normalize the distance matrix if global strategy used
	if ( m_strategy == SPE_GLOBAL )
	{
		float64_t alpha = 1.0 / CMath::max(distance_matrix.matrix, N*N) 
					* CMath::sqrt(2.0);
		CMath::scale_vector(alpha, distance_matrix.matrix, N*N);
	}

	// Compute neighborhood matrix if local strategy used
	SGMatrix< int32_t > neighbors_mat;
	if ( m_strategy == SPE_LOCAL )
	{
		SG_DEBUG("Computing neighborhood matrix\n");
		neighbors_mat = get_neighborhood_matrix(distance_matrix, m_k);
	}

	// Initialize vectors in the embedded space randomly, Y is the short for
	// new_feature_matrix
	SGMatrix< float64_t > Y(m_target_dim, N);
	CMath::random_vector(Y.matrix, m_target_dim*N, 0.0, 1.0);

	// SPE's loop
	
	// Initialize the maximum number of iterations
	int32_t max_iter = 2000 + CMath::round(0.04 * N*N);
	if ( m_strategy == SPE_LOCAL )
		max_iter *= 3;

	// Initialize the learning parameter
	float32_t lambda = 1.0;

	// Initialize variables to use in the main loop
	int32_t i, j, k;
	float64_t sum = 0.0;
	index_t   idx1 = 0, idx2 = 0, idx = 0;

	SGVector< float64_t > scale(m_nupdates);
	SGVector< float64_t > D(m_nupdates);
	SGMatrix< float64_t > Yd(m_nupdates, m_target_dim);
	SGVector< float64_t > Rt(m_nupdates);
	int32_t* ind2 = NULL;

	// Variables required just if local strategy used
	SGMatrix< int32_t > ind1Neighbors;
	SGVector< int32_t > J2;
	if ( m_strategy == SPE_LOCAL )
	{
		ind2 = SG_MALLOC(int32_t, m_nupdates);

		CMath::resize(ind1Neighbors.matrix, 0, m_k*m_nupdates);
		ind1Neighbors.num_rows = m_k;
		ind1Neighbors.num_cols = m_nupdates;

		CMath::resize(J2.vector, 0, m_nupdates);
		J2.vlen = m_nupdates;
	}

	for ( i = 0 ; i < max_iter ; ++i )
	{
		if ( !(i % 1000) )
			SG_DEBUG("SPE's loop, iteration %d of %d\n", i, max_iter);

		// Select the vectors to be updated in this iteration
		int32_t* J = CMath::randperm(N);

		// Pointer to the first set of vector indices to update
		int32_t* ind1 = J;

		// Pointer ind2 to the second set of vector indices to update
		if ( m_strategy == SPE_GLOBAL )
			ind2 = J + m_nupdates;
		else
		{
			// Select the second set of indices to update among neighbors
			// of the first set
				
			// Get the neighbors of interest
			for ( j = 0 ; j < m_nupdates ; ++j )
			{
				for ( k = 0 ; k < m_k ; ++k )
					ind1Neighbors[k + j*m_k] =
						neighbors_mat.matrix[k + ind1[j]*m_k];
			}

			// Generate pseudo-random indices
			for ( j = 0 ; j < m_nupdates ; ++j )
			{
				J2[j] = CMath::round( CMath::random(0.0, 1.0)*(m_k-1) )
						+ m_k*j;
			}

			// Select final indices
			for ( j = 0 ; j < m_nupdates ; ++j )
				ind2[j] = ind1Neighbors.matrix[ J2[j] ];
		}

		// Compute distances betweeen the selected points in embedded space

		for ( j = 0 ; j < m_nupdates ; ++j )
		{
			sum = 0.0;

			for ( k = 0 ; k < m_target_dim ; ++k )
			{
				idx1 = k + ind1[j]*m_target_dim;
				idx2 = k + ind2[j]*m_target_dim;
				sum += CMath::sq(Y.matrix[idx1] - Y.matrix[idx2]);
			}

			D[j] = CMath::sqrt(sum);
		}

		// Get the corresponding distances in the original space
		for ( j = 0 ; j < m_nupdates ; ++j )
			Rt[j] = distance_matrix.matrix[ ind1[j]*N + ind2[j] ];

		// Compute some terms for update

		// Scale factor: (Rt - D) ./ (D + m_tolerance)
		for ( j = 0 ; j < m_nupdates ; ++j )
			scale[j] = ( Rt[j] - D[j] ) / ( D[j] + m_tolerance );

		// Difference matrix: Y(ind1) - Y(ind2)
		for ( j = 0 ; j < m_nupdates ; ++j )
			for ( k = 0 ; k < m_target_dim ; ++k )
			{
				idx1 = k + ind1[j]*m_target_dim;
				idx2 = k + ind2[j]*m_target_dim;
				idx  = k +      j *m_target_dim;

				Yd[idx] = Y[idx1] - Y[idx2];
			}

		// Update the location of the vectors in the embedded space
		for ( j = 0 ; j < m_nupdates ; ++j )
			for ( k = 0 ; k < m_target_dim ; ++k )
			{
				idx1 = k + ind1[j]*m_target_dim;
				idx2 = k + ind2[j]*m_target_dim;
				idx  = k +      j *m_target_dim;

				Y[idx1] += lambda / 2 * scale[j] * Yd[idx];
				Y[idx2] -= lambda / 2 * scale[j] * Yd[idx];
			}

		// Update the learning parameter
		lambda = lambda - ( lambda / max_iter );

		// Free memory
		delete[] J;
	}

	// Free memory
	distance_matrix.destroy_matrix();
	scale.destroy_vector();
	D.destroy_vector();
	Yd.destroy_matrix();
	Rt.destroy_vector();
	if ( m_strategy == SPE_LOCAL )
	{
		ind1Neighbors.destroy_matrix();
		neighbors_mat.destroy_matrix();
		J2.destroy_vector();
		delete[] ind2;
	}
	SG_UNREF(features);

	return (CFeatures*)( new CSimpleFeatures< float64_t >(Y) );
}

SGMatrix<int32_t> CStochasticProximityEmbedding::get_neighborhood_matrix(SGMatrix<float64_t> distance_matrix, int32_t k)
{
	int32_t i;
	int32_t N = distance_matrix.num_rows;

	int32_t* neighborhood_matrix = SG_MALLOC(int32_t, N*k);

	float64_t max_dist = CMath::max(distance_matrix.matrix,N*N);

	CoverTree<SPE_COVERTREE_POINT>* coverTree = new CoverTree<SPE_COVERTREE_POINT>(max_dist);

	for (i=0; i<N; i++)
		coverTree->insert(SPE_COVERTREE_POINT(i,distance_matrix));

	for (i=0; i<N; i++)
	{
		std::vector<SPE_COVERTREE_POINT> neighbors =
		   coverTree->kNearestNeighbors(SPE_COVERTREE_POINT(i,distance_matrix),k+1);
		for (std::size_t m=1; m<unsigned(k+1); m++)
			neighborhood_matrix[i*k+m-1] = neighbors[m].point_index;
	}

	delete coverTree;

	return SGMatrix<int32_t>(neighborhood_matrix,k,N);
}

#endif /* HAVE_LAPACK */
