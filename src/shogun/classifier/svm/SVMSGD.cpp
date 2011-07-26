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

   Shogun adjustments (w) 2008-2009 Soeren Sonnenburg
*/

#include <shogun/classifier/svm/SVMSGD.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

// Available losses
#define HINGELOSS 1
#define SMOOTHHINGELOSS 2
#define SQUAREDHINGELOSS 3
#define LOGLOSS 10
#define LOGLOSSMARGIN 11

// Select loss
#define LOSS SQUAREDHINGELOSS

// One when bias is regularized
#define REGULARIZEBIAS 0

inline
float64_t loss(float64_t z)
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
float64_t dloss(float64_t z)
{
#if LOSS == LOGLOSS
	if (z < 0)
		return 1 / (exp(z) + 1);
	float64_t ez = exp(-z);
	return ez / (ez + 1);
#elif LOSS == LOGLOSSMARGIN
	if (z < 1)
		return 1 / (exp(z-1) + 1);
	float64_t ez = exp(1-z);
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


CSVMSGD::CSVMSGD()
: CLinearMachine()
{
	init();
}

CSVMSGD::CSVMSGD(float64_t C)
: CLinearMachine()
{
	init();

	C1=C;
	C2=C;
}

CSVMSGD::CSVMSGD(float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
	set_labels(trainlab);
}

CSVMSGD::~CSVMSGD()
{
}

bool CSVMSGD::train(CFeatures* data)
{
	// allocate memory for w and initialize everyting w and bias with 0
	ASSERT(labels);

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}

	ASSERT(features);
	ASSERT(labels->is_two_class_labeling());

	int32_t num_train_labels=labels->get_num_labels();
	w_dim=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	ASSERT(num_vec>0);

	SG_FREE(w);
	w=SG_MALLOCX(float64_t, w_dim);
	memset(w, 0, w_dim*sizeof(float64_t));
	bias=0;

	float64_t lambda= 1.0/(C1*num_vec);

	// Shift t in order to have a 
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	float64_t maxw = 1.0 / sqrt(lambda);
	float64_t typw = sqrt(maxw);
	float64_t eta0 = typw / CMath::max(1.0,dloss(-typw));
	t = 1 / (eta0 * lambda);

	SG_INFO("lambda=%f, epochs=%d, eta0=%f\n", lambda, epochs, eta0);


	//do the sgd
	calibrate();

	SG_INFO("Training on %d vectors\n", num_vec);
	CSignal::clear_cancel();

	for(int32_t e=0; e<epochs && (!CSignal::cancel_computations()); e++)
	{
		count = skip;
		for (int32_t i=0; i<num_vec; i++)
		{
			float64_t eta = 1.0 / (lambda * t);
			float64_t y = labels->get_label(i);
			float64_t z = y * (features->dense_dot(i, w, w_dim) + bias);

#if LOSS < LOGLOSS
			if (z < 1)
#endif
			{
				float64_t etd = eta * dloss(z);
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
				float64_t r = 1 - eta * lambda * skip;
				if (r < 0.8)
					r = pow(1 - eta * lambda, skip);
				CMath::scale_vector(r, w, w_dim);
				count = skip;
			}
			t++;
		}
	}

	float64_t wnorm =  CMath::dot(w,w, w_dim);
	SG_INFO("Norm: %.6f, Bias: %.6f\n", wnorm, bias);

	return true;
}

void CSVMSGD::calibrate()
{ 
	ASSERT(features);
	int32_t num_vec=features->get_num_vectors();
	int32_t c_dim=features->get_dim_feature_space();

	ASSERT(num_vec>0);
	ASSERT(c_dim>0);

	float64_t* c=SG_MALLOCX(float64_t, c_dim);
	memset(c, 0, c_dim*sizeof(float64_t));

	SG_INFO("Estimating sparsity and bscale num_vec=%d num_feat=%d.\n", num_vec, c_dim);

	// compute average gradient size
	int32_t n = 0;
	float64_t m = 0;
	float64_t r = 0;

	for (int32_t j=0; j<num_vec && m<=1000; j++, n++)
	{
		r += features->get_nnz_features_for_vector(j);
		features->add_to_dense_vec(1, j, c, c_dim, true);

		//waste cpu cycles for readability
		//(only changed dims need checking)
		m=CMath::max(c, c_dim);
	}

	// bias update scaling
	bscale = 0.5*m/n;

	// compute weight decay skip
	skip = (int32_t) ((16 * n * c_dim) / r);
	SG_INFO("using %d examples. skip=%d  bscale=%.6f\n", n, skip, bscale);

	SG_FREE(c);
}

void CSVMSGD::init()
{
	t=1;
	C1=1;
	C2=1;
	wscale=1;
	bscale=1;
	epochs=5;
	skip=1000;
	count=1000;
	use_bias=true;

	use_regularized_bias=false;

    m_parameters->add(&C1, "C1",  "Cost constant 1.");
    m_parameters->add(&C2, "C2",  "Cost constant 2.");
    m_parameters->add(&wscale, "wscale",  "W scale");
    m_parameters->add(&bscale, "bscale",  "b scale");
    m_parameters->add(&epochs, "epochs",  "epochs");
    m_parameters->add(&skip, "skip",  "skip");
    m_parameters->add(&count, "count",  "count");
    m_parameters->add(&use_bias, "use_bias",  "Indicates if bias is used.");
    m_parameters->add(&use_regularized_bias, "use_regularized_bias",  "Indicates if bias is regularized.");
}
