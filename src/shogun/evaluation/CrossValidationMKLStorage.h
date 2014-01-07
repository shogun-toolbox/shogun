/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 *
 */

#ifndef __CROSSVALIDATIONMKLSTORAGE_H_
#define __CROSSVALIDATIONMKLSTORAGE_H_

#include <evaluation/CrossValidationOutput.h>
#include <lib/SGMatrix.h>

namespace shogun
{

class CMachine;
class CLabels;
class CEvaluation;

/** @brief Class for storing MKL weights in every fold of cross-validation */
class CCrossValidationMKLStorage: public CCrossValidationOutput
{
public:

	/** constructor */
	CCrossValidationMKLStorage() : CCrossValidationOutput() {}

	/** destructor */
	virtual ~CCrossValidationMKLStorage() {};

	/** @return name of SG_SERIALIZABLE */
	virtual const char* get_name() const { return "CrossValidationMKLStorage"; }

	/** update trained machine. Here, stores MKL weights in local matrix
	 *
	 * @param machine trained machine instance
	 * @param prefix prefix for output
	 */
	virtual void update_trained_machine(CMachine* machine,
			const char* prefix="");

	/** @return mkl weights matrix, one set of weights per column,
	 * num_folds*num_runs columns, one fold after another */
	virtual SGMatrix<float64_t> get_mkl_weights() { return m_mkl_weights; }

protected:
	/** storage for MKL weights, one set per column
	 * num_kernel times num_folds*num_runs matrix where all folds of a runs
	 * are added one after another */
	SGMatrix<float64_t> m_mkl_weights;
};

}

#endif /* __CROSSVALIDATIONMKLSTORAGE_H_ */
