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
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/kernel/GaussianKernel.h>

namespace shogun
{

class CFeatures;
class CDistance;
class CKernel;

/** @brief the class DimensionReductionPreprocessor, a base
 * class for preprocessors used to lower the dimensionality of given 
 * simple features (dense matrices). 
 */
class CDimensionReductionPreprocessor: public CSimplePreprocessor<float64_t>
{
public:

	/* constructor */
	CDimensionReductionPreprocessor() : CSimplePreprocessor<float64_t>()
	{
		m_target_dim = 1;
		m_distance = new CEuclidianDistance();
		m_distance->parallel = this->parallel;
		SG_REF(this->parallel);
		m_kernel = new CGaussianKernel();
		m_kernel->parallel = this->parallel;
		SG_REF(this->parallel);

		init();
	};

	/* destructor */
	virtual ~CDimensionReductionPreprocessor() 
	{
		delete m_distance;
		SG_UNREF(this->parallel);
		delete m_kernel;
		SG_UNREF(this->parallel);
	};

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
		ASSERT(dim>0 || dim==AUTO_TARGET_DIM);
		m_target_dim = dim;
	}

	/** getter for target dimension
	 * @return target dimension
	 */
	int32_t inline get_target_dim() const
	{
		return m_target_dim;
	}

	/** setter for distance
	 * @param distance distance to set
	 */
	void inline set_distance(CDistance* distance)
	{
		SG_UNREF(m_distance->parallel);
		SG_UNREF(m_distance);
		SG_REF(distance);
		m_distance = distance;
		m_distance->parallel = this->parallel;
		SG_REF(this->parallel);
	}

	/** setter for kernel
	 * @param kernel kernel to set
	 */
	void inline set_kernel(CKernel* kernel)
	{
		SG_UNREF(m_kernel->parallel);
		SG_UNREF(m_kernel);
		SG_REF(kernel);
		m_kernel = kernel;
		m_kernel->parallel = this->parallel;
		SG_REF(this->parallel);
	}

public:

	/** const indicating target dimensionality should be determined automagically */
	static const int32_t AUTO_TARGET_DIM = -1;

protected:

	virtual int32_t detect_dim(SGMatrix<float64_t> distance_matrix)
	{
		SG_NOTIMPLEMENTED;
		return 0;
	}


	/** default init */
	void init()
	{
		m_parameters->add(&m_target_dim, "target_dim",
		                  "target dimensionality of preprocessor");
		m_parameters->add((CSGObject**)&m_distance, "distance",
		                  "distance to be used for embedding");
		m_parameters->add((CSGObject**)&m_kernel, "kernel",
		                  "kernel to be used for embedding");
	}

protected:

	/** target dim of dimensionality reduction preprocessor */
	int32_t m_target_dim;

	/** distance to be used */
	CDistance* m_distance;

	/** kernel to be used */
	CKernel* m_kernel;

};
}

#endif /* DIMENSIONREDUCTIONPREPROCESSOR_H_ */
