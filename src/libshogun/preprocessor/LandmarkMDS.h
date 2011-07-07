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
 * fast multidimensional scaling using landmark multidimensional
 * scaling algorithm described in
 *  
 * Sparse multidimensional scaling using landmark points
 * V De Silva, J B Tenenbaum (2004) Technology, p. 1-41
 * 
 */
class CLandmarkMDS: public CClassicMDS
{
public:

	/* constructor */
	CLandmarkMDS();

	/* destructor */
	virtual ~CLandmarkMDS();

	/** empty init
	 */
	virtual bool init(CFeatures* data);

	/** empty cleanup
	 */
	virtual void cleanup();

	/** apply preprocessor to CDistance using landmark mds
	 * @param distance (should be approximate euclidean for consistent results)
	 * @return new features with distance similar to given as much as possible
	 */	
	virtual CSimpleFeatures<float64_t>* apply_to_distance(CDistance* distance);

	/** apply preprocessor to feature matrix,
	 * changes feature matrix to the one having target dimensionality
	 * @param features features which feature matrix should be processed
	 * @return new feature matrix
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preprocessor to feature vector
	 * @param vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);


	/** get name */
	virtual inline const char* get_name() const { return "LandmarkMDS"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_LANDMARKMDS; };

	/** set number of landmarks 
	 * should be lesser than number of examples and greater than 3
	 * for consistent embedding
	 * @param num number of landmark to be set
	 */
	void set_landmark_number(int32_t num)
	{
		m_landmark_number = num;
	};

	/** get number of landmarks 
	 * @return current number of landmarks
	 */
	int32_t get_landmark_number()
	{
		return m_landmark_number;
	};

	/** apply preprocessor to CDistance
	 * this method is used internally by other preprocessors
	 * involving landmark MDS (e.g. Landmark Isomap) at some stage
	 * @param distance given distance (should be approximate euclidean for consistent results)
	 * @return new feature matrix representing given distance
	 */
	SGMatrix<float64_t> embed_by_distance(CDistance* distance);

protected:

	/** current number of landmarks */
	int32_t m_landmark_number;	

	/** subroutine used to shuffle count indexes among of total_count ones
	 * with Fisher-Yates (as well as Knuth) shuffle
	 * @param count number of indexes to be shuffled and returned
	 * @param total_count total number of indexes
	 * @return sampled indexes for landmarks
	 */
	SGVector<int32_t> get_landmark_idxs(int32_t count, int32_t total_count);

};

}

#endif /* HAVE_LAPACK */
#endif /* LANDMARK_H_ */
