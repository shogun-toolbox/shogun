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

#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "distance/Distance.h"

namespace shogun
{

class CFeatures;

/** @brief
 *
 */
class CDimensionReductionPreprocessor: public CSimplePreprocessor<float64_t>
{
public:

	/* constructor */
	CDimensionReductionPreprocessor() : CSimplePreprocessor<float64_t>(), m_target_dim(1) {};

	/* destructor */
	virtual ~CDimensionReductionPreprocessor() {};

	/** init
	 *
	 */
	virtual bool init(CFeatures* data)
	{
		return true;
	};

	/** cleanup
	 *
	 */
	virtual void cleanup()
	{

	};

	/** apply preproc to feature matrix
	 *
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features)
	{
		return ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
	};

	/** apply preproc to feature vector
	 *
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector)
	{
		return vector;
	};

	/** get name */
	virtual inline const char* get_name() const { return "DIMREDUCTIONPREPROCESSOR"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_UNKNOWN; };

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

	/** target dim */
	int32_t m_target_dim;

};
}

#endif /* DIMENSIONREDUCTIONPREPROCESSOR_H_ */
