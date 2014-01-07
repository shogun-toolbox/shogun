/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Harshit Syal
 * Copyright (C) 2012 Harshit Syal
 */
#include <lib/config.h>

#ifdef HAVE_LAPACK
#include <classifier/svm/NewtonSVM.h>
#include <mathematics/Math.h>
#include <machine/LinearMachine.h>
#include <features/DotFeatures.h>
#include <labels/Labels.h>
#include <labels/BinaryLabels.h>
#include <mathematics/lapack.h>

//#define DEBUG_NEWTON
//#define V_NEWTON
using namespace shogun;

CNewtonSVM::CNewtonSVM()
: CLinearMachine(), C(1), use_bias(true)
{
}

CNewtonSVM::CNewtonSVM(float64_t c, CDotFeatures* traindat, CLabels* trainlab, int32_t itr)
: CLinearMachine()
{
	lambda=1/c;
	num_iter=itr;
	prec=1e-6;
	num_iter=20;
	use_bias=true;
	C=c;
	set_features(traindat);
	set_labels(trainlab);
}


CNewtonSVM::~CNewtonSVM()
{
}


bool CNewtonSVM::train_machine(CFeatures* data)
{
	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_BINARY)

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

	//Assigning dimensions for whole class scope
	x_n=num_vec;
	x_d=num_feat;

	ASSERT(num_vec==train_labels.vlen)

	float64_t* weights = SG_CALLOC(float64_t, x_d+1);
	float64_t* out=SG_MALLOC(float64_t, x_n);
	SGVector<float64_t>::fill_vector(out, x_n, 1.0);

	int32_t *sv=SG_MALLOC(int32_t, x_n), size_sv=0, iter=0;
	float64_t obj, *grad=SG_MALLOC(float64_t, x_d+1);
	float64_t t;

	while(1)
	{
		iter++;

		if (iter>num_iter)
		{
			SG_PRINT("Maximum number of Newton steps reached. Try larger lambda")
			break;
		}

		obj_fun_linear(weights, out, &obj, sv, &size_sv, grad);

#ifdef DEBUG_NEWTON
		SG_PRINT("fun linear passed !\n")
		SG_PRINT("Obj =%f\n", obj)
		SG_PRINT("Grad=\n")

		for (int32_t i=0; i<x_d+1; i++)
			SG_PRINT("grad[%d]=%.16g\n", i, grad[i])
		SG_PRINT("SV=\n")

		for (int32_t i=0; i<size_sv; i++)
			SG_PRINT("sv[%d]=%d\n", i, sv[i])
#endif

		SGVector<float64_t> sgv;
		float64_t* Xsv = SG_MALLOC(float64_t, x_d*size_sv);
		for (int32_t k=0; k<size_sv; k++)
		{
			sgv=features->get_computed_dot_feature_vector(sv[k]);
			for (int32_t j=0; j<x_d; j++)
				Xsv[k*x_d+j]=sgv.vector[j];
		}
		int32_t tx=x_d;
		int32_t ty=size_sv;
		SGMatrix<float64_t>::transpose_matrix(Xsv, tx, ty);

#ifdef DEBUG_NEWTON
		SGMatrix<float64_t>::display_matrix(Xsv, x_d, size_sv);
#endif

		float64_t* lcrossdiag=SG_MALLOC(float64_t, (x_d+1)*(x_d+1));
		float64_t* vector=SG_MALLOC(float64_t, x_d+1);

		for (int32_t i=0; i<x_d; i++)
			vector[i]=lambda;

		vector[x_d]=0;

		SGMatrix<float64_t>::create_diagonal_matrix(lcrossdiag, vector, x_d+1);
		float64_t* Xsv2=SG_MALLOC(float64_t, x_d*x_d);
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, x_d, x_d, size_sv,
				1.0, Xsv, size_sv, Xsv, size_sv, 0.0, Xsv2, x_d);
		float64_t* sum=SG_CALLOC(float64_t, x_d);

		for (int32_t j=0; j<x_d; j++)
		{
			for (int32_t i=0; i<size_sv; i++)
				sum[j]+=Xsv[i+j*size_sv];
		}

		float64_t* Xsv2sum=SG_MALLOC(float64_t, (x_d+1)*(x_d+1));

		for (int32_t i=0; i<x_d; i++)
		{
			for (int32_t j=0; j<x_d; j++)
				Xsv2sum[j*(x_d+1)+i]=Xsv2[j*x_d+i];

			Xsv2sum[x_d*(x_d+1)+i]=sum[i];
		}

		for (int32_t j=0; j<x_d; j++)
			Xsv2sum[j*(x_d+1)+x_d]=sum[j];

		Xsv2sum[x_d*(x_d+1)+x_d]=size_sv;
		float64_t* identity_matrix=SG_MALLOC(float64_t, (x_d+1)*(x_d+1));

		SGVector<float64_t>::fill_vector(vector, x_d+1, 1.0);

		SGMatrix<float64_t>::create_diagonal_matrix(identity_matrix, vector, x_d+1);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, x_d+1, x_d+1,
				x_d+1, 1.0, lcrossdiag, x_d+1, identity_matrix, x_d+1, 1.0,
				Xsv2sum, x_d+1);

		float64_t* inverse=SG_MALLOC(float64_t, (x_d+1)*(x_d+1));
		int32_t r=x_d+1;
		SGMatrix<float64_t>::pinv(Xsv2sum, r, r, inverse);

		float64_t* step=SG_MALLOC(float64_t, r);
		float64_t* s2=SG_MALLOC(float64_t, r);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, r, 1, r, 1.0,
				inverse, r, grad, r, 0.0, s2, r);

		for (int32_t i=0; i<r; i++)
			step[i]=-s2[i];

		line_search_linear(weights, step, out, &t);

#ifdef DEBUG_NEWTON
		SG_PRINT("t=%f\n\n", t)

		for (int32_t i=0; i<x_n; i++)
			SG_PRINT("out[%d]=%.16g\n", i, out[i])

		for (int32_t i=0; i<x_d+1; i++)
			SG_PRINT("weights[%d]=%.16g\n", i, weights[i])
#endif

		SGVector<float64_t>::vec1_plus_scalar_times_vec2(weights, t, step, r);
		float64_t newton_decrement;
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, r, -0.5,
				step, r, grad, r, 0.0, &newton_decrement, 1);
#ifdef V_NEWTON
		SG_PRINT("Itr=%d, Obj=%f, No of sv=%d, Newton dec=%0.3f, line search=%0.3f\n\n",
				iter, obj, size_sv, newton_decrement, t);
#endif

		SG_FREE(Xsv);
		SG_FREE(vector);
		SG_FREE(lcrossdiag);
		SG_FREE(Xsv2);
		SG_FREE(Xsv2sum);
		SG_FREE(identity_matrix);
		SG_FREE(inverse);
		SG_FREE(step);
		SG_FREE(s2);

		if (newton_decrement*2<prec*obj)
			break;
	}

#ifdef V_NEWTON
	SG_PRINT("FINAL W AND BIAS Vector=\n\n")
	CMath::display_matrix(weights, x_d+1, 1);
#endif

	set_w(SGVector<float64_t>(weights, x_d));
	set_bias(weights[x_d]);

	SG_FREE(sv);
	SG_FREE(grad);
	SG_FREE(out);

	return true;


}

void CNewtonSVM::line_search_linear(float64_t* weights, float64_t* d, float64_t*
		out, float64_t* tx)
{
	SGVector<float64_t> Y=((CBinaryLabels*) m_labels)->get_labels();
	float64_t* outz=SG_MALLOC(float64_t, x_n);
	float64_t* temp1=SG_MALLOC(float64_t, x_n);
	float64_t* temp1forout=SG_MALLOC(float64_t, x_n);
	float64_t* outzsv=SG_MALLOC(float64_t, x_n);
	float64_t* Ysv=SG_MALLOC(float64_t, x_n);
	float64_t* Xsv=SG_MALLOC(float64_t, x_n);
	float64_t* temp2=SG_MALLOC(float64_t, x_d);
	float64_t t=0.0;
	float64_t* Xd=SG_MALLOC(float64_t, x_n);

	for (int32_t i=0; i<x_n; i++)
		Xd[i]=features->dense_dot(i, d, x_d);

	SGVector<float64_t>::add_scalar(d[x_d], Xd, x_n);

#ifdef DEBUG_NEWTON
	CMath::display_vector(d, x_d+1, "Weight vector");

	for (int32_t i=0; i<x_d+1; i++)
		SG_SPRINT("Xd[%d]=%.18g\n", i, Xd[i])

	CMath::display_vector(Xd, x_n, "XD vector=");
#endif

	float64_t wd;
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, x_d, lambda,
			weights, x_d, d, x_d, 0.0, &wd, 1);
	float64_t tempg, dd;
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, x_d, lambda, d,
			x_d, d, x_d, 0.0, &dd, 1);

	float64_t g, h;
	int32_t sv_len=0, *sv=SG_MALLOC(int32_t, x_n);

	do
	{
		SGVector<float64_t>::vector_multiply(temp1, Y.vector, Xd, x_n);
		memcpy(temp1forout, temp1, sizeof(float64_t)*x_n);
		SGVector<float64_t>::scale_vector(t, temp1forout, x_n);
		SGVector<float64_t>::add(outz, 1.0, out, -1.0, temp1forout, x_n);

		// Calculation of sv
		sv_len=0;

		for (int32_t i=0; i<x_n; i++)
		{
			if (outz[i]>0)
				sv[sv_len++]=i;
		}

		//Calculation of gradient 'g'
		for (int32_t i=0; i<sv_len; i++)
		{
			outzsv[i]=outz[sv[i]];
			Ysv[i]=Y.vector[sv[i]];
			Xsv[i]=Xd[sv[i]];
		}

		memset(temp1, 0, sizeof(float64_t)*sv_len);
		SGVector<float64_t>::vector_multiply(temp1, outzsv, Ysv, sv_len);
		tempg=SGVector<float64_t>::dot(temp1, Xsv, sv_len);
		g=wd+(t*dd);
		g-=tempg;

		// Calculation of second derivative 'h'
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, sv_len, 1.0,
				Xsv, sv_len, Xsv, sv_len, 0.0, &h, 1);
		h+=dd;

		// Calculation of 1D Newton step 'd'
		t-=g/h;

		if (((g*g)/h)<1e-10)
			break;

	} while(1);

	for (int32_t i=0; i<x_n; i++)
		out[i]=outz[i];
	*tx=t;

	SG_FREE(sv);
	SG_FREE(temp1);
	SG_FREE(temp2);
	SG_FREE(temp1forout);
	SG_FREE(outz);
	SG_FREE(outzsv);
	SG_FREE(Ysv);
	SG_FREE(Xsv);
	SG_FREE(Xd);
}

void CNewtonSVM::obj_fun_linear(float64_t* weights, float64_t* out,
		float64_t* obj, int32_t* sv, int32_t* numsv, float64_t* grad)
{
	SGVector<float64_t> v=((CBinaryLabels*) m_labels)->get_labels();

	for (int32_t i=0; i<x_n; i++)
	{
		if (out[i]<0)
			out[i]=0;
	}

#ifdef DEBUG_NEWTON
	for (int32_t i=0; i<x_n; i++)
		SG_SPRINT("out[%d]=%.16g\n", i, out[i])
#endif

	//create copy of w0
	float64_t* w0=SG_MALLOC(float64_t, x_d+1);
	memcpy(w0, weights, sizeof(float64_t)*(x_d));
	w0[x_d]=0; //do not penalize b

	//create copy of out
	float64_t* out1=SG_MALLOC(float64_t, x_n);

	//compute steps for obj
	SGVector<float64_t>::vector_multiply(out1, out, out, x_n);
	float64_t p1=SGVector<float64_t>::sum(out1, x_n)/2;
	float64_t C1;
	float64_t* w0copy=SG_MALLOC(float64_t, x_d+1);
	memcpy(w0copy, w0, sizeof(float64_t)*(x_d+1));
	SGVector<float64_t>::scale_vector(0.5, w0copy, x_d+1);
	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, 1, x_d+1, lambda,
			w0, x_d+1, w0copy, x_d+1, 0.0, &C1, 1);
	*obj=p1+C1;
	SGVector<float64_t>::scale_vector(lambda, w0, x_d);
	float64_t* temp=SG_CALLOC(float64_t, x_n); //temp = out.*Y
	SGVector<float64_t>::vector_multiply(temp, out, v.vector, x_n);
	float64_t* temp1=SG_CALLOC(float64_t, x_d);
	SGVector<float64_t> vec;

	for (int32_t i=0; i<x_n; i++)
	{
		features->add_to_dense_vec(temp[i], i, temp1, x_d);
#ifdef DEBUG_NEWTON
		SG_SPRINT("\ntemp[%d]=%f", i, temp[i])
		CMath::display_vector(vec.vector, x_d, "vector");
		CMath::display_vector(temp1, x_d, "debuging");
#endif
	}
	float64_t* p2=SG_MALLOC(float64_t, x_d+1);

	for (int32_t i=0; i<x_d; i++)
		p2[i]=temp1[i];

	p2[x_d]=SGVector<float64_t>::sum(temp, x_n);
	SGVector<float64_t>::add(grad, 1.0, w0, -1.0, p2, x_d+1);
	int32_t sv_len=0;

	for (int32_t i=0; i<x_n; i++)
	{
		if (out[i]>0)
			sv[sv_len++]=i;
	}

	*numsv=sv_len;

	SG_FREE(w0);
	SG_FREE(w0copy);
	SG_FREE(out1);
	SG_FREE(temp);
	SG_FREE(temp1);
	SG_FREE(p2);
}
#endif //HAVE_LAPACK
