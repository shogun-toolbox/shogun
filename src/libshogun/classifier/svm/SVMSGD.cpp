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

#include "SVMSGD.h"
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
#define LOSS HINGELOSS

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
: COnlineLinearMachine()
{
	init();
}

CSVMSGD::CSVMSGD(float64_t C)
: COnlineLinearMachine()
{
	init();

	C1=C;
	C2=C;
}

CSVMSGD::CSVMSGD(float64_t C, CStreamingDotFeatures* traindat)
: COnlineLinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
}

CSVMSGD::~CSVMSGD()
{
}

bool CSVMSGD::train(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			SG_ERROR("Specified features are not of type CStreamingDotFeatures\n");
		set_features((CStreamingDotFeatures*) data);
	}

	features->start_parser();
	
	// allocate memory for w and initialize everyting w and bias with 0
	ASSERT(features);
	ASSERT(features->get_has_labels());
	if (w)
		delete[] w;
	w_dim=1;
	w=new float64_t;
	bias=0;

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
	if (features->is_seekable())
		features->reset_stream();

	CSignal::clear_cancel();
	
	int32_t vec_count=0;
	for(int32_t e=0; e<epochs && (!CSignal::cancel_computations()); e++)
	{
		vec_count=0;
		count = skip;
		while (features->get_next_example())
		{
			vec_count++;
			// Expand w vector if more features are seen
			features->expand_if_required(w, w_dim);
				
			float64_t eta = 1.0 / (lambda * t);
			float64_t y = features->get_label();
			float64_t z = y * (features->dense_dot(w, w_dim) + bias);

#if LOSS < LOGLOSS
			if (z < 1)
#endif
			{
				float64_t etd = eta * dloss(z);
				features->add_to_dense_vec(etd * y / wscale, w, w_dim);

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

			features->release_example();
		}

		// If the stream is seekable, reset the stream to the first example (for epochs > 1)
		if (features->is_seekable() && e < epochs-1)
			features->reset_stream();
		else
			break;

	}

	features->end_parser();
	float64_t wnorm =  CMath::dot(w,w, w_dim);
	SG_INFO("Norm: %.6f, Bias: %.6f\n", wnorm, bias);

	return true;
}

void CSVMSGD::calibrate(int32_t max_vec_num)
{ 
	int32_t c_dim=1;
	float64_t* c=new float64_t;
	
	// compute average gradient size
	int32_t n = 0;
	float64_t m = 0;
	float64_t r = 0;

	int32_t counter=0;
	while (features->get_next_example() && m<=1000)
	{
		features->expand_if_required(c, c_dim);
			
		r += features->get_nnz_features_for_vector();
		features->add_to_dense_vec(1, c, c_dim, true);

		//waste cpu cycles for readability
		//(only changed dims need checking)
		m=CMath::max(c, c_dim);
		n++;

		features->release_example();
		if (n>=max_vec_num)
			break;
	}

	// bias update scaling
	bscale = m/n;

	// compute weight decay skip
	skip = (int32_t) ((16 * n * c_dim) / r);

	SG_INFO("using %d examples. skip=%d  bscale=%.6f\n", n, skip, bscale);

	delete[] c;
}

void CSVMSGD::init()
{
	t=1;
	C1=1;
	C2=1;
	lambda=1e-4;
	wscale=1;
	bscale=1;
	epochs=1;
	skip=1000;
	count=1000;
	use_bias=true;

	use_regularized_bias=false;

	m_parameters->add(&C1, "C1",  "Cost constant 1.");
	m_parameters->add(&C2, "C2",  "Cost constant 2.");
	m_parameters->add(&lambda, "lambda", "Regularization parameter.");
	m_parameters->add(&wscale, "wscale",  "W scale");
	m_parameters->add(&bscale, "bscale",  "b scale");
	m_parameters->add(&epochs, "epochs",  "epochs");
	m_parameters->add(&skip, "skip",  "skip");
	m_parameters->add(&count, "count",  "count");
	m_parameters->add(&use_bias, "use_bias",  "Indicates if bias is used.");
	m_parameters->add(&use_regularized_bias, "use_regularized_bias",  "Indicates if bias is regularized.");
}
