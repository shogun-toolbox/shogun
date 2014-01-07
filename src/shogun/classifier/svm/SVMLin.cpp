/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <classifier/svm/SVMLin.h>
#include <labels/Labels.h>
#include <mathematics/Math.h>
#include <lib/external/ssl.h>
#include <machine/LinearMachine.h>
#include <features/DotFeatures.h>
#include <labels/Labels.h>
#include <labels/BinaryLabels.h>

using namespace shogun;

CSVMLin::CSVMLin()
: CLinearMachine(), C1(1), C2(1), epsilon(1e-5), use_bias(true)
{
}

CSVMLin::CSVMLin(
	float64_t C, CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine(), C1(C), C2(C), epsilon(1e-5), use_bias(true)
{
	set_features(traindat);
	set_labels(trainlab);
}


CSVMLin::~CSVMLin()
{
}

bool CSVMLin::train_machine(CFeatures* data)
{
	ASSERT(m_labels)

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}

	ASSERT(features)

	SGVector<float64_t> train_labels=((CBinaryLabels*) m_labels)->get_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==train_labels.vlen)

	struct options Options;
	struct data Data;
	struct vector_double Weights;
	struct vector_double Outputs;

	Data.l=num_vec;
	Data.m=num_vec;
	Data.u=0;
	Data.n=num_feat+1;
	Data.nz=num_feat+1;
	Data.Y=train_labels.vector;
	Data.features=features;
	Data.C = SG_MALLOC(float64_t, Data.l);

	Options.algo = SVM;
	Options.lambda=1/(2*get_C1());
	Options.lambda_u=1/(2*get_C1());
	Options.S=10000;
	Options.R=0.5;
	Options.epsilon = get_epsilon();
	Options.cgitermax=10000;
	Options.mfnitermax=50;
	Options.Cp = get_C2()/get_C1();
	Options.Cn = 1;

	if (use_bias)
		Options.bias=1.0;
	else
		Options.bias=0.0;

	for (int32_t i=0;i<num_vec;i++)
	{
		if(train_labels.vector[i]>0)
			Data.C[i]=Options.Cp;
		else
			Data.C[i]=Options.Cn;
	}
	ssl_train(&Data, &Options, &Weights, &Outputs);
	ASSERT(Weights.vec && Weights.d==num_feat+1)

	float64_t sgn=train_labels.vector[0];
	for (int32_t i=0; i<num_feat+1; i++)
		Weights.vec[i]*=sgn;

	set_w(SGVector<float64_t>(Weights.vec, num_feat));
	set_bias(Weights.vec[num_feat]);

	SG_FREE(Data.C);
	SG_FREE(Outputs.vec);
	return true;
}
