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

/** @brief CDiffusionMaps used to preprocess given data using 
  * diffusion maps dimensionality reduction technique
  */
class CDiffusionMaps: public CDimensionReductionPreprocessor<float64_t>
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
	void set_t(int32_t t);

	/** getter for t parameter
	 * @return t value
	 */
	int32_t get_t() const;

	/** get name */
	virtual const char* get_name() const;

	/** get type */
	virtual EPreprocessorType get_type() const;

protected:

	/** default init */
	void init();

protected:

	/** number of steps */
	int32_t m_t;

};
}

#endif /* HAVE_LAPACK */
#endif /* DIFFUSIONMAPS_H_ */
