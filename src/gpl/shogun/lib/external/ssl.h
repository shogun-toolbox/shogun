/*    Copyright 2006 Vikas Sindhwani (vikass@cs.uchicago.edu)
	  SVM-lin: Fast SVM Solvers for Supervised and Semi-supervised Learning

	  This file is part of SVM-lin.

	  SVM-lin is free software; you can redistribute it and/or modify
	  it under the terms of the GNU General Public License as published by
	  the Free Software Foundation; either version 2 of the License, or
	  (at your option) any later version.

	  SVM-lin is distributed in the hope that it will be useful,
	  but WITHOUT ANY WARRANTY; without even the implied warranty of
	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	  GNU General Public License for more details.

	  You should have received a copy of the GNU General Public License
	  along with SVM-lin (see gpl.txt); if not, write to the Free Software
	  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
	  */
#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef _SSL_H
#define _SSL_H

/* OPTIMIZATION CONSTANTS */
#define CGITERMAX 10000 /* maximum number of CGLS iterations */
#define SMALL_CGITERMAX 10 /* for heuristic 1 in reference [2] */
#define EPSILON   1e-6 /* most tolerances are set to this value */
#define BIG_EPSILON 0.01 /* for heuristic 2 in reference [2] */
#define RELATIVE_STOP_EPS 1e-9 /* for L2-SVM-MFN relative stopping criterion */
#define MFNITERMAX 50 /* maximum number of MFN iterations */
#define TSVM_ANNEALING_RATE 1.5 /* rate at which lambda_u is increased in TSVM */
#define TSVM_LAMBDA_SMALL 1e-5 /* lambda_u starts from this value */
#define DA_ANNEALING_RATE 1.5 /* annealing rate for DA */
#define DA_INIT_TEMP 10 /* initial temperature relative to lambda_u */
#define DA_INNER_ITERMAX 100 /* maximum fixed temperature iterations for DA */
#define DA_OUTER_ITERMAX 30 /* maximum number of outer loops for DA */

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{
/** Data: Input examples are stored in sparse (Compressed Row Storage) format */
struct data
{
	/** number of examples */
	int32_t m;
	/** number of labeled examples */
	int32_t l;
	/** number of unlabeled examples l+u = m */
	int32_t u;
	/** number of features */
	int32_t n;
	/** number of non-zeros */
	int32_t nz;

	/** features */
	shogun::CDotFeatures* features;
	/** labels */
	float64_t *Y;
	/** cost associated with each example */
	float64_t *C;
};

/** defines a vector of doubles */
struct vector_double
{
	/** number of elements */
	int32_t d;
	/** ptr to vector elements*/
	float64_t *vec;
};

/** defines a vector of ints for index subsets */
struct vector_int
{
	/** number of elements */
	int32_t d;
	/** ptr to vector elements */
	int32_t *vec;
};

enum { RLS, SVM, TSVM, DA_SVM }; /* currently implemented algorithms */

/** various options user + internal optimisation */
struct options
{
	/* user options */
	/** regularization parameter */
	int32_t algo;
	/** regularization parameter */
	float64_t lambda;
	/** regularization parameter over unlabeled examples */
	float64_t lambda_u;
	/** maximum number of TSVM switches per fixed-weight label optimization */
	int32_t S;
	/** expected fraction of unlabeled examples in positive class */
	float64_t R;
	/** cost for positive examples */
	float64_t Cp;
	/** cost for negative examples */
	float64_t Cn;

	/*  internal optimization options */
	/** all tolerances */
	float64_t epsilon;
	/** max iterations for CGLS */
	int32_t cgitermax;
	/** max iterations for L2_SVM_MFN */
	int32_t mfnitermax;

	/** 1.0 if bias is to be used, 0.0 otherwise */
	float64_t bias;
};

/** used in line search */
class Delta {
	public:
		/** default constructor */
		Delta() { delta=0.0; index=0;s=0; }

		/** delta */
		float64_t delta;
		/** index */
		int32_t index;
		/** s */
		int32_t s;
};

inline bool operator<(const Delta& a , const Delta& b)
{
	return (a.delta < b.delta);
}

void initialize_ssl(struct vector_double *A, int32_t k, float64_t a);
/* initializes a vector_double to be of length k, all elements set to a */
void initialize_ssl(struct vector_int *A, int32_t k);
/* initializes a vector_int to be of length k, elements set to 1,2..k. */
void GetLabeledData(struct data *Data_Labeled, const struct data *Data);
/* extracts labeled data from Data and copies it into Data_Labeled */
float64_t norm_square(const vector_double *A); /* returns squared length of A */

/* ssl_train: takes data, options, uninitialized weight and output
   vector_doubles, routes it to the algorithm */
/* the learnt weight vector and the outputs it gives on the data matrix are saved */
void ssl_train(
	struct data *Data,
	struct options *Options,
	struct vector_double *W, /* weight vector */
	struct vector_double *O); /* output vector */

/* svmlin algorithms and their subroutines */

/* Conjugate Gradient for Sparse Linear Least Squares Problems */
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_{i in Subset} Data->C[i] (Y[i]- w' x_i)^2 */
/* over a subset of examples x_i specified by vector_int Subset */
int32_t CGLS(
	const struct data *Data,
	const struct options *Options,
	const struct vector_int *Subset,
	struct vector_double *Weights,
	struct vector_double *Outputs);

/* Linear Modified Finite Newton L2-SVM*/
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_i Data->C[i] max(0,1 - Y[i] w' x_i)^2 */
int32_t L2_SVM_MFN(
	const struct data *Data,
	struct options *Options,
	struct vector_double *Weights,
	struct vector_double *Outputs,
	int32_t ini); /* use ini=0 if no good starting guess for Weights, else 1 */

float64_t line_search(
	float64_t *w,
	float64_t *w_bar,
	float64_t lambda,
	float64_t *o,
	float64_t *o_bar,
	float64_t *Y,
	float64_t *C,
	int32_t d,
	int32_t l);

/* Transductive L2-SVM */
/* Solves : min_(w, Y[i],i in UNlabeled) 0.5*Options->lamda*w'*w + 0.5*(1/Data->l)*sum_{i in labeled} max(0,1 - Y[i] w' x_i)^2 + 0.5*(Options->lambda_u/Data->u)*sum_{i in UNlabeled} max(0,1 - Y[i] w' x_i)^2
   subject to: (1/Data->u)*sum_{i in UNlabeled} max(0,Y[i]) = Options->R */
int32_t TSVM_MFN(
	const struct data *Data,
	struct options *Options,
	struct vector_double *Weights,
	struct vector_double *Outputs);

int32_t switch_labels(
	float64_t* Y,
	float64_t* o,
	int32_t* JU,
	int32_t u,
	int32_t S);

/* Deterministic Annealing*/
int32_t DA_S3VM(
	struct data *Data,
	struct options *Options,
	struct vector_double *Weights,
	struct vector_double *Outputs);

void optimize_p(
	const float64_t* g, int32_t u, float64_t T, float64_t r, float64_t*p);

int32_t optimize_w(
	const struct data *Data,
	const  float64_t *p,
	struct options *Options,
	struct vector_double *Weights,
	struct vector_double *Outputs,
	int32_t ini);

float64_t transductive_cost(
	float64_t normWeights,
	float64_t *Y,
	float64_t *Outputs,
	int32_t m,
	float64_t lambda,
	float64_t lambda_u);

float64_t entropy(const  float64_t *p, int32_t u);

/* KL-divergence */
float64_t KL(const  float64_t *p, const  float64_t *q, int32_t u);
}
#endif // _SSL_H

#endif // DOXYGEN_SHOULD_SKIP_THIS
