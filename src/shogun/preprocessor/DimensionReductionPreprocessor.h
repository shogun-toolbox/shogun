/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef DIMENSIONREDUCTIONPREPROCESSOR_H_
#define DIMENSIONREDUCTIONPREPROCESSOR_H_

#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;

/** @brief the abstract class DimensionReductionPreprocessor, a base
 * class for preprocessors used to lower the dimensionality of given 
 * simple features (dense matrices). 
 */
class CDimensionReductionPreprocessor: public CSimplePreprocessor<float64_t>
{
public:

	/* constructor */
	CDimensionReductionPreprocessor() : CSimplePreprocessor<float64_t>(), m_target_dim(1) {};

	/* destructor */
	virtual ~CDimensionReductionPreprocessor() {};

	/** init
	 * set true by default, should be defined if dimension reduction
	 * preprocessor is using some initialization
	 */
	virtual bool init(CFeatures* data)
	{
		return true;
	};

	/** cleanup
	 * set empty by default, should be defined if dimension reduction
	 * preprocessor should free some resources
	 */
	virtual void cleanup()
	{

	};

	/** apply preproc to feature matrix
	 * by default does nothing, returns given features' matrix
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features)
	{
		return ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
	};

	/** apply preproc to feature vector
	 * by default does nothing, returns given feature vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector)
	{
		return vector;
	};

	/** get name */
	virtual inline const char* get_name() const { return "DIMREDUCTIONPREPROCESSOR"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_DIMENSIONREDUCTIONPREPROCESSOR; };

	/** setter for target dimension
	 * @param dim target dimension
	 */
	void inline set_target_dim(int32_t dim)
	{
		ASSERT(dim>0);
		m_target_dim = dim;
	}

	/** getter for target dimension
	 * @return target dimension
	 */
	int32_t inline get_target_dim()
	{
		return m_target_dim;
	}

protected:

	/** target dim of dimensionality reduction preprocessor */
	int32_t m_target_dim;

};
}

#endif /* DIMENSIONREDUCTIONPREPROCESSOR_H_ */
