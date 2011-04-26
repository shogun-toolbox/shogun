/*
   SVM with Quasi-Newton stochastic gradient
   Copyright (C) 2009- Antoine Bordes

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

   Shogun adjustments (w) 2011 Siddharth Kherada
*/

#include "classifier/svm/SGDQN.h"
#include "base/Parameter.h"
#include "lib/Signal.h"
#include "lib/Mathematics.h"
#include "lib/Loss.h"


using namespace shogun;

CSGDQN::CSGDQN()
: CLinearClassifier()
{
	init();
}

CSGDQN::CSGDQN(float64_t C)
: CLinearClassifier()
{
	init();

	C1=C;
	C2=C;
}

CSGDQN::CSGDQN(float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearClassifier()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
	set_labels(trainlab);
}

CSGDQN::~CSGDQN()
{
}

void CSGDQN::compute_ratio(float64_t* W,float64_t* W_1,float64_t* B,float64_t* dst,int32_t dim,float64_t lambda,float64_t loss)
{
	for (int32_t i=0; i < dim;i++)
	{
		float64_t diffw=W_1[i]-W[i];
		if(diffw)
			B[i]+=diffw/ (lambda*diffw+ loss*dst[i]);
		else
			B[i]+=1/lambda;
	}
}

void CSGDQN::combine_and_clip(float64_t* Bc,float64_t* B,int32_t dim,float64_t c1,float64_t c2,float64_t v1,float64_t v2)
{
	for (int32_t i=0; i < dim;i++)
	{
		if(B[i])
		{
			Bc[i] = Bc[i] * c1 + B[i] * c2;
			Bc[i]= CMath::min(CMath::max(Bc[i],v1),v2);
		}
	}
}

bool CSGDQN::train(CFeatures* data)
{

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

	delete[] w;
	w=new float64_t[w_dim];
	memset(w, 0, w_dim*sizeof(float64_t));

	float64_t lambda= 1.0/(C1*num_vec);

	// Shift t in order to have a
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	float64_t maxw = 1.0 / sqrt(lambda);
	float64_t typw = sqrt(maxw);
	float64_t eta0 = typw / CMath::max(1.0,CLoss::dloss(-typw));
	t = 1 / (eta0 * lambda);

	SG_INFO("lambda=%f, epochs=%d, eta0=%f\n", lambda, epochs, eta0);


	float64_t* Bc=new float64_t[w_dim];
	CMath::fill_vector(Bc, w_dim, 1/lambda);

	float64_t* result=new float64_t[w_dim];
	float64_t* B=new float64_t[w_dim];
	float64_t* w_1=new float64_t[w_dim];

	//Calibrate
	calibrate();

	SG_INFO("Training on %d vectors\n", num_vec);
	CSignal::clear_cancel();

	for(int32_t e=0; e<epochs && (!CSignal::cancel_computations()); e++)
	{
		count = skip;
		bool updateB=false;
		for (int32_t i=0; i<num_vec; i++)
		{
			float64_t* dst;
			int32_t alen;
			features->get_feature_vector(&dst,&alen,i);
			ASSERT(w_dim==alen);
			float64_t eta = 1.0/t;
			float64_t y = labels->get_label(i);
			float64_t z = y * features->dense_dot(i, w, w_dim);
			if(updateB==true)
			{
#if LOSS < LOGLOSS
				if (z < 1)
#endif
				{
					w_1=w;
					float64_t loss_1=CLoss::dloss(z);
					CMath::vector_multiply(result,Bc,dst,w_dim);
					CMath::add(w,eta*loss_1*y,result,1.0,w,w_dim);
					float64_t z2 = y * features->dense_dot(i, w, w_dim);
					float64_t diffloss = CLoss::dloss(z2) - loss_1;
					if(diffloss)
					{
						compute_ratio(w,w_1,B,dst,w_dim,lambda,y*diffloss);
						if(t>skip)
							combine_and_clip(Bc,B,w_dim,(t-skip)/(t+skip),2*skip/(t+skip),1/(100*lambda),100/lambda);
						else
							combine_and_clip(Bc,B,w_dim,t/(t+skip),skip/(t+skip),1/(100*lambda),100/lambda);
					}
				}
				updateB=false;
			}
			else
			{
				if(--count<=0)
				{
					CMath::vector_multiply(result,Bc,w,w_dim);
					CMath::add(w,-skip*lambda*eta,result,1.0,w,w_dim);
					count = skip;
					updateB=true;
				}
#if LOSS < LOGLOSS
				if (z < 1)
#endif
				{
					CMath::vector_multiply(result,Bc,dst,w_dim);
					CMath::add(w,eta*CLoss::dloss(z)*y,result,1.0,w,w_dim);
				}
			}
			t++;
		}
	}
	delete[] result;
	delete[] w_1;
	delete[] B;

	return true;
}



void CSGDQN::calibrate()
{
	ASSERT(features);
	int32_t num_vec=features->get_num_vectors();
	int32_t c_dim=features->get_dim_feature_space();

	ASSERT(num_vec>0);
	ASSERT(c_dim>0);

	SG_INFO("Estimating sparsity num_vec=%d num_feat=%d.\n", num_vec, c_dim);

	int32_t n = 0;
	float64_t r = 0;

	for (int32_t j=0; j<num_vec ; j++, n++)
		r += features->get_nnz_features_for_vector(j);


	// compute weight decay skip
	skip = (int32_t) ((16 * n * c_dim) / r);
}

void CSGDQN::init()
{
	t=0;
	C1=1;
	C2=1;
	epochs=5;
	skip=1000;
	count=1000;

	m_parameters->add(&C1, "C1",  "Cost constant 1.");
	m_parameters->add(&C2, "C2",  "Cost constant 2.");
	m_parameters->add(&epochs, "epochs",  "epochs");
	m_parameters->add(&skip, "skip",  "skip");
	m_parameters->add(&count, "count",  "count");
}
