/*
 * Copyright (c) 2000-2009 Chih-Chung Chang and Chih-Jen Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. Neither name of copyright holders nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Shogun specific adjustments (w) 2006-2009 Soeren Sonnenburg
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef _LIBSVM_H
#define _LIBSVM_H

#include <shogun/lib/config.h>

#include <shogun/kernel/Kernel.h>

namespace shogun
{
/** SVM node */
struct svm_node
{
	/** index */
	int32_t index;
};

/** SVM problem */
struct svm_problem
{


public:

	/** default constructor */
	svm_problem() {
		y = NULL;
		x = NULL;
		C = NULL;
		pv = NULL;
	}


	/** l */
	int32_t l;
	/** y */
	float64_t *y;
	/** SVM node x */
	struct svm_node **x;
	/** weight per example */
    float64_t *C;
    /** precomputed p */
	float64_t *pv;

};

enum { C_SVC=1, NU_SVC=2, NU_MULTICLASS_SVC=3, ONE_CLASS=4, EPSILON_SVR=5, NU_SVR=6 };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

/** SVM parameter */
struct svm_parameter
{
	/** SVM type */
	int32_t svm_type;
	/** kernel type */
	int32_t kernel_type;
	/** kernel */
	shogun::CKernel* kernel;
	/** for poly */
	int32_t degree;
	/** for poly/rbf/sigmoid */
	float64_t gamma;
	/** for poly/sigmoid */
	float64_t coef0;

	/* these are for training only */
	/** in MB */
	float64_t cache_size;
	/** maximum training time */
	float64_t max_train_time;
	/** stopping criteria */
	float64_t eps;
	/** for C_SVC, EPSILON_SVR and NU_SVR */
	float64_t C;
	/** number of classes */
	int32_t nr_class;
	/** for C_SVC */
	int32_t nr_weight;
	/** for C_SVC */
	int32_t *weight_label;
	/** for C_SVC */
	float64_t* weight;
	/** for NU_SVC, ONE_CLASS, and NU_SVR */
	float64_t nu;
	/** for EPSILON_SVR */
	float64_t p;
	/** use the shrinking heuristics */
	int32_t shrinking;
	/** compute bias */
	bool use_bias;
};

/** svm_model */
struct svm_model
{
	/** parameter */
	svm_parameter param;
	/** number of classes, = 2 in regression/one class svm */
	int32_t nr_class;
	/** total #SV */
	int32_t l;
	/** SVs (SV[l]) */
	svm_node **SV;
	/** coefficients for SVs in decision functions (sv_coef[n-1][l]) */
	float64_t **sv_coef;
	/** constants in decision functions (rho[n*(n-1)/2]) */
	float64_t *rho;
	/** norm_wcw */
	float64_t *normwcw;

	// for classification only

	/** label of each class (label[n]) */
	int32_t *label;
	/** number of SVs for each class (nSV[n])
	 * nSV[0] + nSV[1] + ... + nSV[n-1] = l
	 */
	int32_t *nSV;
	// XXX
	/** 1 if svm_model is created by svm_load_model
	    0 if svm_model is created by svm_train
	*/
	int32_t free_sv;
	/** objective */
	float64_t objective;
};



struct svm_model *svm_train(
	const struct svm_problem *prob, const struct svm_parameter *param);

float64_t svm_predict(const struct svm_model *model, const struct svm_node *x);

void svm_destroy_model(struct svm_model *model);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
}
#endif /* _LIBSVM_H */

#endif // DOXYGEN_SHOULD_SKIP_THIS
