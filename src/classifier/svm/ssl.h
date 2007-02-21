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

#include "lib/common.h"
#include "features/SparseFeatures.h"

/* Data: Input examples are stored in sparse (Compressed Row Storage) format */
struct data 
{
	int m; /* number of examples */
	int l; /* number of labeled examples */
	int u; /* number of unlabeled examples l+u = m */
	int n; /* number of features */ 
	int nz; /* number of non-zeros */

	CSparseFeatures<DREAL>* features;
	//double *val; /* data values (nz elements) [CRS format] */
	//int *rowptr; /* n+1 vector [CRS format] */
	//int *colind; /* nz elements [CRS format] */ 
	double *Y;   /* labels */
	double *C;   /* cost associated with each example */
};

struct vector_double /* defines a vector of doubles */
{
	int d; /* number of elements */
	double *vec; /* ptr to vector elements*/
};

struct vector_int /* defines a vector of ints for index subsets */
{
	int d; /* number of elements */
	int *vec; /* ptr to vector elements */
};

enum { RLS, SVM, TSVM, DA_SVM }; /* currently implemented algorithms */

struct options 
{
	/* user options */
	int algo; /* 1 to 4 for RLS,SVM,TSVM,DASVM */
	double lambda; /* regularization parameter */
	double lambda_u; /* regularization parameter over unlabeled examples */
	int S; /* maximum number of TSVM switches per fixed-weight label optimization */
	double R; /* expected fraction of unlabeled examples in positive class */
	double Cp; /* cost for positive examples */
	double Cn; /* cost for negative examples */
	/*  internal optimization options */    
	double epsilon; /* all tolerances */
	int cgitermax;  /* max iterations for CGLS */
	int mfnitermax; /* max iterations for L2_SVM_MFN */

};

class Delta { /* used in line search */
	public: 
		Delta() {delta=0.0; index=0;s=0;};  
		double delta;   
		int index;
		int s;   
};
inline bool operator<(const Delta& a , const Delta& b) { return (a.delta < b.delta);};

void initialize(struct vector_double *A, int k, double a);  
/* initializes a vector_double to be of length k, all elements set to a */
void initialize(struct vector_int *A, int k); 
/* initializes a vector_int to be of length k, elements set to 1,2..k. */
//void GetLabeledData(struct data *Data_Labeled, const struct data *Data); 
/* extracts labeled data from Data and copies it into Data_Labeled */
void Write(const char *file_name, const struct vector_double *somevector);
/* writes a vector into filename, one element per line */
void Clear(struct data *a); /* deletes a */
void Clear(struct vector_double *a); /* deletes a */
void Clear(struct vector_int *a); /* deletes a */
double norm_square(const vector_double *A); /* returns squared length of A */

/* ssl_train: takes data, options, uninitialized weight and output
   vector_doubles, routes it to the algorithm */
/* the learnt weight vector and the outputs it gives on the data matrix are saved */
void ssl_train(struct data *Data, 
		struct options *Options,
		struct vector_double *W, /* weight vector */
		struct vector_double *O); /* output vector */

/* svmlin algorithms and their subroutines */

/* Conjugate Gradient for Sparse Linear Least Squares Problems */
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_{i in Subset} Data->C[i] (Y[i]- w' x_i)^2 */
/* over a subset of examples x_i specified by vector_int Subset */
int CGLS(const struct data *Data, 
		const struct options *Options, 
		const struct vector_int *Subset,
		struct vector_double *Weights,
		struct vector_double *Outputs);

/* Linear Modified Finite Newton L2-SVM*/
/* Solves: min_w 0.5*Options->lamda*w'*w + 0.5*sum_i Data->C[i] max(0,1 - Y[i] w' x_i)^2 */
int L2_SVM_MFN(const struct data *Data, 
		struct options *Options, 
		struct vector_double *Weights,
		struct vector_double *Outputs,
		int ini); /* use ini=0 if no good starting guess for Weights, else 1 */
double line_search(double *w, 
		double *w_bar,
		double lambda,
		double *o, 
		double *o_bar, 
		double *Y, 
		double *C,
		int d,
		int l);

/* Transductive L2-SVM */
/* Solves : min_(w, Y[i],i in UNlabeled) 0.5*Options->lamda*w'*w + 0.5*(1/Data->l)*sum_{i in labeled} max(0,1 - Y[i] w' x_i)^2 + 0.5*(Options->lambda_u/Data->u)*sum_{i in UNlabeled} max(0,1 - Y[i] w' x_i)^2 
   subject to: (1/Data->u)*sum_{i in UNlabeled} max(0,Y[i]) = Options->R */
//int   TSVM_MFN(const struct data *Data, 
//		struct options *Options, 
//		struct vector_double *Weights,
//		struct vector_double *Outputs);
int switch_labels(double* Y, double* o, int* JU, int u, int S);

/* Deterministic Annealing*/
int DA_S3VM(struct data *Data, 
		struct options *Options, 
		struct vector_double *Weights,
		struct vector_double *Outputs);
void optimize_p(const double* g, int u, double T, double r, double*p);
int optimize_w(const struct data *Data, 
		const  double *p,
		struct options *Options, 
		struct vector_double *Weights,
		struct vector_double *Outputs,
		int ini);
double transductive_cost(double normWeights,double *Y, double *Outputs, int m, double lambda,double lambda_u);
double entropy(const  double *p, int u); 
double KL(const  double *p, const  double *q, int u); /* KL-divergence */

#endif
