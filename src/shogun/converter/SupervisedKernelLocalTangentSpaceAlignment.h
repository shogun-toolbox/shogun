/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef SUPERVISEDKERNELLOCALTANGENTSPACEALIGNMENT_H_
#define SUPERVISEDKERNELLOCALTANGENTSPACEALIGNMENT_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/KernelLocalTangentSpaceAlignment.h>
#include <shogun/features/Features.h>
#include <shogun/features/Labels.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief class LocalTangentSpaceAlignment (part of the
 * Efficient Dimensionality Reduction Toolkit) used to embed
 * data using supervised kernel extension of the Local Tangent Space
 * Alignment (LTSA) algorithm.
 */
class CSupervisedKernelLocalTangentSpaceAlignment: public CKernelLocalTangentSpaceAlignment
{
public:

	/** constructor */
	CSupervisedKernelLocalTangentSpaceAlignment();

	/** constructor with kernel parameter
	 * @param kernel kernel to be used
	 */
	CSupervisedKernelLocalTangentSpaceAlignment(CKernel* kernel, CLabels* labels);

	/** set labels */
	void set_labels(CLabels* labels)
	{
		SG_REF(labels);
		SG_UNREF(m_labels);
		m_labels = labels;
	}
	/** get labels */
	CLabels* get_labels() const
	{
		SG_REF(m_labels);
		return m_labels;
	}

	/** destructor */
	virtual ~CSupervisedKernelLocalTangentSpaceAlignment();

	/** get name */
	virtual const char* get_name() const;

protected:

	/** constructs neighborhood matrix by kernel matrix
	 * @param kernel_matrix kernel matrix to be used
	 * @param k k
	 * @return matrix containing indexes of neighbors of i-th object
	 * in i-th column
	 */
	virtual SGMatrix<int32_t> get_neighborhood_matrix(SGMatrix<float64_t> kernel_matrix, int32_t k);

protected:

	/** labels */
	CLabels* m_labels;

};
}

#endif /* HAVE_LAPACK */
#endif /* SUPERVISEDKERNELLOCALTANGENTSPACEALINGMENT_H_ */
