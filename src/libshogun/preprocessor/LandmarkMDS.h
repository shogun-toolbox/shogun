/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef LANDMARKMDS_H_
#define LANDMARKMDS_H_
#ifdef HAVE_LAPACK
#include "preprocessor/SimplePreprocessor.h"
#include "preprocessor/ClassicMDS.h"
#include "features/Features.h"
#include "distance/Distance.h"
#include "distance/CustomDistance.h"

namespace shogun
{

class CFeatures;

class CDistance;

/** @brief class LandmarkMDS used to perform
 *  fast multidimensional scaling using landmark multidimensional
 *  scaling algorithm described in
 *
 *
 */
class CLandmarkMDS: public CClassicMDS
{
public:

	/* constructor */
	CLandmarkMDS();

	/* destructor */
	virtual ~CLandmarkMDS();

	/** init
	 * @param data feature vectors for preproc
	 */
	virtual bool init(CFeatures* data);

	/** cleanup
	 *
	 */
	virtual void cleanup();


	/** apply preproc to distance
	 *
	 */
	virtual CSimpleFeatures<float64_t>* apply_to_distance(CDistance* distance);

	/** apply preproc to feature matrix
	 *
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preproc to feature vector
	 *
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);


	/** get name */
	virtual inline const char* get_name() const { return "LandmarkMDS"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LANDMARKMDS; };

	/** set number of landmarks */
	void set_landmark_number(int32_t num)
	{
		m_landmark_number = num;
	};

	/** get number of landmarks */
	int32_t get_landmark_number()
	{
		return m_landmark_number;
	};

	/** apply preproc to distance
	 *
	 */
	SGMatrix<float64_t> embed_by_distance(CDistance* distance);

protected:

	/** number of landmarks */
	int32_t m_landmark_number;	

	/**
	 * @return sampled indexes for landmarks
	 */
	SGVector<int32_t> get_landmark_idxs(int32_t count, int32_t total_count);

};

}

#endif /* HAVE_LAPACK */
#endif /* LANDMARK_H_ */
