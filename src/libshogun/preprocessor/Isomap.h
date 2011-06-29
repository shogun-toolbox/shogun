/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef ISOMAP_H_
#define ISOMAP_H_
#ifdef HAVE_LAPACK
#include "preprocessor/DimensionReductionPreprocessor.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "features/Features.h"
#include "distance/Distance.h"
#include "distance/CustomDistance.h"
#include "distance/EuclidianDistance.h"

namespace shogun
{

class CFeatures;

class CDistance;

enum EIsomapType
{
	KISOMAP,
	EISOMAP
};

/** @brief the class Isomap
 *
 */
class CIsomap: public CDimensionReductionPreprocessor
{
public:

	/* constructor */
	CIsomap() :
		CDimensionReductionPreprocessor(), m_epsilon(1.0),
		m_k(3), m_type(EISOMAP) {};

	/* destructor */
	virtual ~CIsomap() {};

	/** init
	 * @param data feature vectors for preproc
	 */
	virtual bool init(CFeatures* features)
	{
		return true;
	};

	/** cleanup
	 *
	 */
	virtual void cleanup() {};

	/** apply preproc to distance
	 *
	 */
	virtual CSimpleFeatures<float64_t>* apply_to_distance(CDistance* distance)
	{
		CDistance* geodesic_distance = isomap_distance(distance);

		SGMatrix<float64_t> new_features = mds_embed(geodesic_distance);

		delete geodesic_distance;

		return new CSimpleFeatures<float64_t>(new_features);
	}

	/** apply preproc to feature matrix
	 *
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features)
	{
		CSimpleFeatures<float64_t>* simple_features =
				(CSimpleFeatures<float64_t>*) features;
		CDistance* euclidian_distance = new CEuclidianDistance();
		euclidian_distance->init(simple_features,simple_features);
		CDistance* geodesic_distance = isomap_distance(euclidian_distance);

		SGMatrix<float64_t> new_features = mds_embed(geodesic_distance);

		delete geodesic_distance;
		delete euclidian_distance;

		simple_features->set_feature_matrix(new_features);

		return new_features;
	}

	/** apply preproc to feature vector
	 *
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector)
	{
		SG_NOTIMPLEMENTED;
		return vector;
	}

	/** get name */
	virtual inline const char* get_name() const { return "Isomap"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_ISOMAP; };

	/** setter for type of isomap
	 * @param type
	 */
	void inline set_type(EIsomapType type)
	{
		m_type = type;
	}

	/** getter for type of isomap
	 * @return current type
	 */
	EIsomapType inline get_type()
	{
		return m_type;
	}


	/** setter for epsilon parameter
	 * @param epsilon
	 */
	void inline set_epsilon(float64_t epsilon)
	{
		m_epsilon = epsilon;
	}

	/** getter for epsilon parameter
	 * @return epsilon value
	 */
	int32_t inline get_epsilon()
	{
		return m_epsilon;
	}

	/** setter for k parameter
	 * @param k
	 */
	void inline set_k(int32_t k)
	{
		m_k = k;
	}

	/** getter for k parameter
	 * @return k value
	 */
	int32_t inline get_k()
	{
		return m_k;
	}

protected:

	/** epsilon */
	float64_t m_epsilon;

	/** k */
	int32_t m_k;

	/** type of Isomap */
	EIsomapType m_type;

	/** approx geodesic distance */
	CCustomDistance* isomap_distance(CDistance* distance)
	{
		int32_t N,k,i,j;
		float64_t* D_matrix;
		distance->get_distance_matrix(&D_matrix,&N,&N);

		if (m_type==EISOMAP)
		{
			// just replace distances >e with infty
			for (i=0; i<N*N; i++)
			{
				if (D_matrix[i]>m_epsilon)
					D_matrix[i] = CMath::ALMOST_INFTY;
			}
		}
		if (m_type==KISOMAP)
		{
			// cut by k-nearest neighbors
			float64_t* col = new float64_t[N];
			int32_t* col_idx = new int32_t[N];

			for (i=0; i<N; i++)
			{
				for (j=0; j<N; j++)
				{
					col[j] = D_matrix[j*N+i];
					col_idx[j] = j;
				}

				CMath::qsort_index(col,col_idx,N);

				//CMath::display_vector(col_idx,N,"Idx");

				for (j=m_k+1; j<N; j++)
				{
					D_matrix[j*N+col_idx[i]] = CMath::ALMOST_INFTY;
				}
			}

			delete[] col;
			delete[] col_idx;
		}

		// Floyd-Warshall on distance matrix
		for (k=0; k<N; k++)
		{
			for (i=0; i<N; i++)
			{
				for (j=0; j<N; j++)
					D_matrix[i*N+j] =
							CMath::min(D_matrix[i*N+j],
									   D_matrix[i*N+k] + D_matrix[k*N+j]);
			}
		}

		//CMath::display_matrix(D_matrix,N,N,"D");

		CCustomDistance* geodesic_distance = new CCustomDistance(D_matrix,N,N);

		// should be removed if custom distance doesn't copy the matrix
		delete[] D_matrix;

		return geodesic_distance;
	}

	/** mds embedding */
	virtual SGMatrix<float64_t> mds_embed(CDistance* distance) = 0;

};
}
#endif /* HAVE_LAPACK */
#endif /* ISOMAP_H_ */
