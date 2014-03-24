/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifndef __MKLREGRESSION_H__
#define __MKLREGRESSION_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/classifier/mkl/MKL.h>

namespace shogun
{
/** @brief Multiple Kernel Learning for regression
 *
 * Performs support vector regression while learning kernel weights at the same
 * time. Makes only sense if multiple kernels are used.
 *
 * \sa CMKL
 */
class CMKLRegression : public CMKL
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_REGRESSION);

		/** Constructor
		 *
		 * @param s SVM to use as constraint generator in MKL SILP
		 */
		CMKLRegression(CSVM* s=NULL);

		/** Destructor
		 */
		virtual ~CMKLRegression();

		/** compute beta independent term from objective, e.g., in 2-class MKL
		 * sum_i alpha_i etc
		 */
		virtual float64_t compute_sum_alpha();

		/** @return object name */
		virtual const char* get_name() const { return "MKLRegression"; }

	protected:
		/** check run before starting training (to e.g. check if labeling is
		 * two-class labeling in classification case
		 */
		virtual void init_training();

		/** get classifier type
		 *
		 * @return classifier type MKL_REGRESSION
		 */
		virtual EMachineType get_classifier_type() { return CT_MKLREGRESSION; }

		/** compute mkl dual objective
		 *
		 * @return computed dual objective
		 */
		virtual float64_t compute_mkl_dual_objective();
};
}
#endif //__MKLREGRESSION_H__
