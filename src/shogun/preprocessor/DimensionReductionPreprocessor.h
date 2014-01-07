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

#include <preprocessor/DensePreprocessor.h>
#include <converter/EmbeddingConverter.h>
#include <features/Features.h>
#include <distance/Distance.h>

namespace shogun
{

class CFeatures;
class CDistance;
class CKernel;

/** @brief the class DimensionReductionPreprocessor, a base
 * class for preprocessors used to lower the dimensionality of given
 * simple features (dense matrices).
 */
class CDimensionReductionPreprocessor: public CDensePreprocessor<float64_t>
{
public:

	/** default constructor */
	CDimensionReductionPreprocessor();

	/** convenience constructor converting any embeddingconverter into a
	 * dimensionreduction preprocessor
	 *
	 * @param converter embedding converter
	 */
	CDimensionReductionPreprocessor(CEmbeddingConverter* converter);

	/** destructor */
	virtual ~CDimensionReductionPreprocessor();

	/** init
	 * set true by default, should be defined if dimension reduction
	 * preprocessor is using some initialization
	 */
	virtual bool init(CFeatures* data);

	/** cleanup
	 * set empty by default, should be defined if dimension reduction
	 * preprocessor should free some resources
	 */
	virtual void cleanup();

	/** apply preproc to feature matrix
	 * by default does nothing, returns given features' matrix
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

	/** apply preproc to feature vector
	 * by default does nothing, returns given feature vector
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector)
	{
		return vector;
	}

	/** get name */
	virtual const char* get_name() const { return "DimensionReductionPreprocessor"; };

	/** get type */
	virtual EPreprocessorType get_type() const;

	/** setter for target dimension
	 * @param dim target dimension
	 */
	void set_target_dim(int32_t dim);

	/** getter for target dimension
	 * @return target dimension
	 */
	int32_t get_target_dim() const;

	/** setter for distance
	 * @param distance distance to set
	 */
	void set_distance(CDistance* distance);

	/** getter for distance
	 * @return distance
	 */
	CDistance* get_distance() const;

	/** setter for kernel
	 * @param kernel kernel to set
	 */
	void set_kernel(CKernel* kernel);

	/** getter for kernel
	 * @return kernel
	 */
	CKernel* get_kernel() const;

protected:

	/** default init */
	void init();

protected:

	/** target dim of dimensionality reduction preprocessor */
	int32_t m_target_dim;

	/** distance to be used */
	CDistance* m_distance;

	/** kernel to be used */
	CKernel* m_kernel;

	/** embedding converter to be used */
	CEmbeddingConverter* m_converter;
};
}

#endif /* DIMENSIONREDUCTIONPREPROCESSOR_H_ */
