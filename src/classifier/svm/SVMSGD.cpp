/*
   SVM with stochastic gradient
   Copyright (C) 2007- Leon Bottou
   
   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
   $Id: svmsgd.cpp,v 1.13 2007/10/02 20:40:06 cvs Exp $

   Shogun adjustments (w) 2008 Soeren Sonnenburg
*/

#include "classifier/svm/SVMSGD.h"

// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Select loss
#define LOSS HINGELOSS

// One when bias is regularized
#define REGULARIZEBIAS 0

inline
double loss(double z)
{
#if LOSS == LOGLOSS
	if (z >= 0)
		return log(1+exp(-z));
	else
		return -z + log(1+exp(z));
#elif LOSS == LOGLOSSMARGIN
	if (z >= 1)
		return log(1+exp(1-z));
	else
		return 1-z + log(1+exp(z-1));
#elif LOSS == SMOOTHHINGELOSS
	if (z < 0)
		return 0.5 - z;
	if (z < 1)
		return 0.5 * (1-z) * (1-z);
	return 0;
#elif LOSS == SQUAREDHINGELOSS
	if (z < 1)
		return 0.5 * (1 - z) * (1 - z);
	return 0;
#elif LOSS == HINGELOSS
	if (z < 1)
		return 1 - z;
	return 0;
#else
# error "Undefined loss"
#endif
}

inline
double dloss(double z)
{
#if LOSS == LOGLOSS
	if (z < 0)
		return 1 / (exp(z) + 1);
	double ez = exp(-z);
	return ez / (ez + 1);
#elif LOSS == LOGLOSSMARGIN
	if (z < 1)
		return 1 / (exp(z-1) + 1);
	double ez = exp(1-z);
	return ez / (ez + 1);
#elif LOSS == SMOOTHHINGELOSS
	if (z < 0)
		return 1;
	if (z < 1)
		return 1-z;
	return 0;
#elif LOSS == SQUAREDHINGELOSS
	if (z < 1)
		return (1 - z);
	return 0;
#else
	if (z < 1)
		return 1;
	return 0;
#endif
}



CSVMSGD::CSVMSGD(double C)
: CSparseLinearClassifier(), t(1), C1(C), C2(C),
	wscale(1), bscale(1), epochs(5), skip(1000), count(1000), use_bias(true),
	use_regularized_bias(false)
{
}

CSVMSGD::CSVMSGD(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab)
: CSparseLinearClassifier(), t(1), C1(C), C2(C), wscale(1), bscale(1),
	epochs(5), skip(1000), count(1000), use_bias(true),
	use_regularized_bias(false)
{
	w=NULL;
	set_features(traindat);
	set_labels(trainlab);
}

CSVMSGD::~CSVMSGD()
{
	delete[] w;
	w=NULL;
}

bool CSVMSGD::train()
{
	// allocate memory for w and initialize everyting w and bias with 0
	ASSERT(labels);
	ASSERT(get_features());
	ASSERT(labels->is_two_class_labeling());

	INT num_train_labels=labels->get_num_labels();
	w_dim=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	ASSERT(num_vec>0);

	delete[] w;
	w=new DREAL[w_dim];
	memset(w, 0, w_dim*sizeof(DREAL));
	bias=0;

	DREAL lambda= 1.0/(C1*num_vec);

	// Shift t in order to have a 
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	DREAL maxw = 1.0 / sqrt(lambda);
	DREAL typw = sqrt(maxw);
	DREAL eta0 = typw / CMath::max(1.0,dloss(-typw));
	t = 1 / (eta0 * lambda);

	SG_INFO("lambda=%f, epochs=%d, eta0=%f\n", lambda, epochs, eta0);


	//do the sgd
	calibrate();

	SG_INFO("Training on %d vectors\n", num_vec);
	for(INT e=0; e<epochs; e++)
	{
		count = skip;
		for (INT i=0; i<num_vec; i++)
		{
			DREAL eta = 1.0 / (lambda * t);
			DREAL y = labels->get_label(i);
			DREAL z = y * features->dense_dot(1.0, i, w, w_dim, bias);

#if LOSS < LOGLOSS
			if (z < 1)
#endif
			{
				DREAL etd = eta * dloss(z);
				features->add_to_dense_vec(etd * y / wscale, i, w, w_dim);

				if (use_bias)
				{
					if (use_regularized_bias)
						bias *= 1 - eta * lambda * bscale;
					bias += etd * y * bscale;
				}
			}

			if (--count <= 0)
			{
				DREAL r = 1 - eta * lambda * skip;
				if (r < 0.8)
					r = pow(1 - eta * lambda, skip);
				CMath::scale_vector(r, w, w_dim);
				count = skip;
			}
			t++;
		}
	}

	DREAL wnorm =  CMath::dot(w,w, w_dim);
	SG_INFO("Norm: %.6f, Bias: %.6f\n", wnorm, bias);

	return true;
}

void CSVMSGD::calibrate()
{ 
	ASSERT(get_features());
	INT num_vec=features->get_num_vectors();
	INT c_dim=features->get_num_features();

	ASSERT(num_vec>0);
	ASSERT(c_dim>0);

	DREAL* c=new DREAL[c_dim];
	memset(c, 0, c_dim*sizeof(DREAL));

	SG_INFO("Estimating sparsity and bscale num_vec=%d num_feat=%d.\n", num_vec, c_dim);

	// compute average gradient size
	INT n = 0;
	DREAL m = 0;
	DREAL r = 0;

	for (INT j=0; j<num_vec && m<=1000; j++, n++)
	{
		r += features->get_num_sparse_vec_features(j);
		features->add_to_dense_vec(1, j, c, c_dim, true);

		//waste cpu cycles for readability
		//(only changed dims need checking)
		m=CMath::max(c, c_dim);
	}

	// bias update scaling
	bscale = m/n;

	// compute weight decay skip
	skip = (INT) ((16 * n * c_dim) / r);
	SG_INFO("using %d examples. skip=%d  bscale=%.6f\n", n, skip, bscale);

	delete[] c;
}

