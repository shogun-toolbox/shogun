/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef DIFFUSIONMAPS_H_
#define DIFFUSIONMAPS_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/distance/Distance.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief */
class CDiffusionMaps: public CDimensionReductionPreprocessor
{
public:

	/** constructor */
	CDiffusionMaps();

	/** destructor */
	virtual ~CDiffusionMaps();

	/** init
	 * @param features
	 */
	virtual bool init(CFeatures* features);

	/** cleanup
	 */
	virtual void cleanup();

	/** apply preprocessor to features
	 * @param features
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preprocessor to feature vector, not supported for LLE
	 * @param vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** setter for t parameter
	 * @param t t value
	 */
	void inline set_t(float64_t t)
	{
		m_t = t;
	}

	/** getter for t parameter
	 * @return t value
	 */
	int32_t inline get_t()
	{
		return m_t;
	}

	/** get name */
	virtual inline const char* get_name() const { return "DiffusionMaps"; };

	/** get type */
	virtual inline EPreprocessorType get_type() const { return P_DIFFUSIONMAPS; };

protected:

	/** default init */
	void init();

protected:

	/** steps */
	int32_t m_t;

};
}

#endif /* HAVE_LAPACK */
#endif /* DIFFUSIONMAPS_H_ */
