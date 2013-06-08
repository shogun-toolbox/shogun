/*
 * Copyright (c) 2007-2009 The LIBLINEAR Project.
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
 */
#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include <shogun/lib/config.h>

#include <shogun/optimization/liblinear/tron.h>
#include <shogun/features/DotFeatures.h>
#include <vector>

namespace shogun
{

#ifdef __cplusplus
extern "C" {
#endif

/** problem */
struct liblinear_problem
{
	/** l */
	int32_t l;
	/** n */
	int32_t n;
	/** y */
	float64_t* y;
	/** sparse features x */
	CDotFeatures* x;
	/** if bias shall be used */
	bool use_bias;
};

/** parameter */
struct liblinear_parameter
{
	/** solver type */
	int32_t solver_type;

	/* these are for training only */
	/** stopping criteria */
	float64_t eps;
	/** C */
	float64_t C;
	/** number of weights */
	int32_t nr_weight;
	/** weight label */
	int32_t *weight_label;
	/** weight */
	float64_t* weight;
};

/** model */
struct liblinear_model
{
	/** parameter */
	struct liblinear_parameter param;
	/** number of classes */
	int32_t nr_class;
	/** number of features */
	int32_t nr_feature;
	/** w */
	float64_t *w;
	/** label of each class (label[n]) */
	int32_t *label;
	/** bias */
	float64_t bias;
};

void destroy_model(struct liblinear_model *model_);
void destroy_param(struct liblinear_parameter *param);
#ifdef __cplusplus
}
#endif

/** class l2loss_svm_vun */
class l2loss_svm_fun : public function
{
public:
	/** constructor
	 *
	 * @param prob prob
	 * @param Cp Cp
	 * @param Cn Cn
	 */
	l2loss_svm_fun(const liblinear_problem *prob, float64_t Cp, float64_t Cn);
	~l2loss_svm_fun();

	/** fun
	 *
	 * @param w w
	 * @return something floaty
	 */
	float64_t fun(float64_t *w);

	/** grad
	 *
	 * @param w w
	 * @param g g
	 */
	void grad(float64_t *w, float64_t *g);

	/** Hv
	 *
	 * @param s s
	 * @param Hs Hs
	 */
	void Hv(float64_t *s, float64_t *Hs);

	/** get number of variables
	 *
	 * @return number of variables
	 */
	int32_t get_nr_variable();

private:
	void Xv(float64_t *v, float64_t *Xv);
	void subXv(float64_t *v, float64_t *Xv);
	void subXTv(float64_t *v, float64_t *XTv);

	float64_t *C;
	float64_t *z;
	float64_t *D;
	int32_t *I;
	int32_t sizeI;
	const liblinear_problem *prob;
};

/** class l2r_lr_fun */
class l2r_lr_fun : public function
{
public:
	/** constructor
	 *
	 * @param prob prob
	 * @param Cp Cp
	 * @param Cn Cn
	 */
	l2r_lr_fun(const liblinear_problem *prob, float64_t* C);
	~l2r_lr_fun();

	/** fun
	 *
	 * @param w w
	 * @return something floaty
	 */
	float64_t fun(float64_t *w);

	/** grad
	 *
	 * @param w w
	 * @param g g
	 */
	void grad(float64_t *w, float64_t *g);

	/** Hv
	 *
	 * @param s s
	 * @param Hs Hs
	 */
	void Hv(float64_t *s, float64_t *Hs);

	int32_t get_nr_variable();

private:
	void Xv(float64_t *v, float64_t *Xv);
	void XTv(float64_t *v, float64_t *XTv);

	float64_t *C;
	float64_t *z;
	float64_t *D;
	const liblinear_problem *m_prob;
};

class l2r_l2_svc_fun : public function
{
public:
	l2r_l2_svc_fun(const liblinear_problem *prob, float64_t* Cs);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable();

protected:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const liblinear_problem *m_prob;
};

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const liblinear_problem *prob, double *Cs, double p);

	double fun(double *w);
	void grad(double *w, double *g);

private:
	double m_p;
};


struct mcsvm_state
{
	double* w;
	double* B;
	double* G;
	double* alpha;
	double* alpha_new;
	int* index;
	double* QD;
	int* d_ind;
	double* d_val;
	int* alpha_index;
	int* y_index;
	int* active_size_i;
	bool allocated,inited;

	mcsvm_state()
	{
		w = NULL;
		B = NULL;
		G = NULL;
		alpha = NULL;
		alpha_new = NULL;
		index = NULL;
		QD = NULL;
		d_ind = NULL;
		d_val = NULL;
		alpha_index = NULL;
		y_index = NULL;
		active_size_i = NULL;
		allocated = false;
		inited = false;
	}

	~mcsvm_state()
	{
		SG_FREE(w);
		SG_FREE(B);
		SG_FREE(G);
		SG_FREE(alpha);
		SG_FREE(alpha_new);
		SG_FREE(index);
		SG_FREE(QD);
		SG_FREE(d_ind);
		SG_FREE(d_val);
		SG_FREE(alpha_index);
		SG_FREE(y_index);
		SG_FREE(active_size_i);
	}
};

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const liblinear_problem *prob, int nr_class, double *C,
		                double *w0, double eps, int max_iter,
		                double train_time, mcsvm_state* given_state);
		~Solver_MCSVM_CS();
		void solve();
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *C;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		double max_train_time;
		double* w0;
		const liblinear_problem *prob;
		mcsvm_state* state;
};


}
#endif //_LIBLINEAR_H

#endif // DOXYGEN_SHOULD_SKIP_THIS
