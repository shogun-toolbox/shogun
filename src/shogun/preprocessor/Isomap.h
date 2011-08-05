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
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/distance/EuclidianDistance.h>

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief the base class Isomap used to preprocess data using Classic
 * or Landmark K-Isomap. The description is given in
 * 
 * Global versus local methods in nonlinear dimensionality reduction
 * Vin De Silva, Joshua B Tenenbaum (2003)
 * Advances in Neural Information Processing Systems 15 15 (Figure 2) p.721-728
 */
class CIsomap: public CDimensionReductionPreprocessor
{
public:

	/* constructor */
	CIsomap() : CDimensionReductionPreprocessor(), m_k(3) {};

	/* destructor */
	virtual ~CIsomap() {};

	/** empty init
	 */
	virtual bool init(CFeatures* features) { return true; };

	/** empty cleanup
	 */
	virtual void cleanup() {};

	/** apply preprocessor to CDistance using
	 * Isomap of specified type
	 * @param distance
	 * @return new features with distance similar to geodesic
	 */
	virtual CSimpleFeatures<float64_t>* apply_to_distance(CDistance* distance)
	{
		CDistance* geodesic_distance = isomap_distance(distance);

		SGMatrix<float64_t> new_features = mds_embed(geodesic_distance);

		delete geodesic_distance;

		return new CSimpleFeatures<float64_t>(new_features);
	}

	/** apply preprocessor to feature matrix using 
	 * Isomap of specified type
	 * @param features
	 * @return new feature matrix with distance similar to geodesic
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

	/** apply preprocessor to feature vector
	 * @param vector
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

	/** setter for k parameter
	 * @param k
	 */
	void inline set_k(int32_t k)
	{
		ASSERT(k>=1);
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

	/** k, number of neighbors for K-Isomap */
	int32_t m_k;

	/** approx geodesic distance 
	 * @param distance
	 * @return custom distance with approximate geodesic distance matrix
	 */
	CCustomDistance* isomap_distance(CDistance* distance);

	/** mds embedding 
	 * @param distance
	 * @return new feature matrix representing given distance
	 */
	virtual SGMatrix<float64_t> mds_embed(CDistance* distance) = 0;

};
}
#endif /* HAVE_LAPACK */

#endif /* ISOMAP_H_ */
