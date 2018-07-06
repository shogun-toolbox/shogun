/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Harshit Syal, Giovanni De Toni, Michele Mazzoni, 
 *          Viktor Gal, Weijie Lin, Sergey Lisitsyn, Sanuj Sharma
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/base/progress.h>
#include <shogun/classifier/svm/NewtonSVM.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

//#define DEBUG_NEWTON
//#define V_NEWTON
using namespace shogun;

CNewtonSVM::CNewtonSVM() : CIterativeMachine<CLinearMachine>()
{
	lambda = 1;
	m_max_iterations = 20;
	prec = 1e-6;
	C = 1;
	t=0;
	set_compute_bias(true);
}

CNewtonSVM::CNewtonSVM(
    float64_t c, CDotFeatures* traindat, CLabels* trainlab, int32_t itr)
    : CIterativeMachine<CLinearMachine>()
{
	lambda=1/c;
	num_iter=itr;
	prec=1e-6;
	C=c;
	t=0;
	set_features(traindat);
	set_labels(trainlab);
	set_compute_bias(true);
}


CNewtonSVM::~CNewtonSVM()
{
}


void CNewtonSVM::init_model(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}

	ASSERT(features)

	SGVector<float64_t> train_labels = binary_labels(m_labels)->get_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	//Assigning dimensions for whole class scope
	x_n=num_vec;
	x_d=num_feat;

	ASSERT(num_vec==train_labels.vlen)

	SGVector<float64_t>weights(x_d);
	set_w(weights);
	out = SGVector<float64_t>(x_n);
	out.set_const(1.0);
	weights.set_const(0.0);
	
	sv = SGVector<int32_t>(x_n);
	sv.set_const(0);
	grad = SGVector<float64_t>(x_d+1);
	grad.set_const(0.0);
}

void CNewtonSVM::iteration()
{
	obj_fun_linear();
	SGVector<float64_t> weights = get_w();

		SGVector<float64_t> sgv;
		SGMatrix<float64_t> Xsv(x_d, size_sv);
		for (int32_t k=0; k<size_sv; k++)
		{
			sgv=features->get_computed_dot_feature_vector(sv[k]);
			for (int32_t j=0; j<x_d; j++)
				Xsv[k*x_d+j]=sgv.vector[j];
		}
		Xsv = linalg::transpose_matrix(Xsv);

		SGMatrix<float64_t> lcrossdiag(x_d+1, x_d+1);
		SGVector<float64_t> vector(x_d+1);

		for (int32_t i=0; i<x_d; i++)
			vector[i]=lambda;

		vector[x_d]=0;

		SGMatrix<float64_t>::create_diagonal_matrix(lcrossdiag.matrix, vector.vector, x_d+1);
		SGMatrix<float64_t> Xsv2(x_d, x_d);
		linalg::matrix_prod(Xsv, Xsv, Xsv2, true);
		
		float64_t* sum=SG_CALLOC(float64_t, x_d);

		for (int32_t j=0; j<x_d; j++)
		{
			for (int32_t i=0; i<size_sv; i++)
				sum[j]+=Xsv[i+j*size_sv];
		}

		SGMatrix<float64_t> Xsv2sum(x_d+1, x_d+1);

		for (int32_t i=0; i<x_d; i++)
		{
			for (int32_t j=0; j<x_d; j++)
				Xsv2sum[j*(x_d+1)+i]=Xsv2[j*x_d+i];

			Xsv2sum[x_d*(x_d+1)+i]=sum[i];
		}

		for (int32_t j=0; j<x_d; j++)
			Xsv2sum[j*(x_d+1)+x_d]=sum[j];

		Xsv2sum[x_d*(x_d+1)+x_d]=size_sv;
		SGMatrix<float64_t> identity_matrix(x_d+1, x_d+1);
		vector.set_const(1.0);

		SGMatrix<float64_t>::create_diagonal_matrix(identity_matrix.matrix, vector.vector, x_d+1);
		
		auto xsv2sum = linalg::matrix_prod(lcrossdiag, identity_matrix);
		
		linalg::add(Xsv2sum, xsv2sum, Xsv2sum);		
		SGMatrix<float64_t> inverse((x_d+1), (x_d+1));
		SGMatrix<float64_t>::pinv(Xsv2sum.matrix, x_d+1, x_d+1, inverse.matrix);
		
		SGVector<float64_t> step(x_d+1);
		SGVector<float64_t>s2(x_d+1);
		s2 = linalg::matrix_prod(inverse, grad);

		for (int32_t i=0; i<x_d+1; i++)
			step[i]=-s2[i];

	line_search_linear(step);

		SGVector<float64_t> tmp_step(step.data(), x_d, false);
		linalg::add(weights, tmp_step, weights, 1.0, t);
		bias += t*step[x_d];
		float64_t newton_decrement = -0.5*linalg::dot(step, grad);

		if (newton_decrement*2<prec*obj)
		m_complete=true;
}

void CNewtonSVM::line_search_linear(const SGVector<float64_t> d)
{
	SGVector<float64_t> Y = binary_labels(m_labels)->get_labels();
	SGVector<float64_t> outz(x_n);
	SGVector<float64_t> temp1(x_n);
	SGVector<float64_t> temp1forout(x_n);
	SGVector<float64_t> outzsv(x_n);
	SGVector<float64_t> Ysv(x_n);
	SGVector<float64_t> Xsv(x_n);
	SGVector<float64_t> Xd(x_n);
	SGVector<float64_t> weights = get_w();
	for (int32_t i=0; i<x_n; i++)
		Xd[i]=features->dense_dot(i, d.data(), x_d);

	linalg::add_scalar(Xd, d[x_d]);
	
	SGVector<float64_t> tmp_d = SGVector<float64_t>(d.data(), x_d, false);
	float64_t wd = lambda*linalg::dot(weights, tmp_d);
	float64_t dd = lambda*linalg::dot(tmp_d, tmp_d);

	float64_t g, h;
	int32_t sv_len=0;
	SGVector<int32_t> sv(x_n);
	
	do
	{
		// FIXME:: port it to linalg::
		SGVector<float64_t>::vector_multiply(temp1.vector, Y.vector, Xd.vector, x_n);
		sg_memcpy(temp1forout.vector, temp1.vector, sizeof(float64_t)*x_n);
		linalg::scale(temp1forout, temp1forout, t);
		outz = linalg::add(out, temp1forout, 1.0, -1.0);

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
			Ysv[i]=Y[sv[i]];
			Xsv[i]=Xd[sv[i]];
		}

		memset(temp1.vector, 0, sizeof(float64_t)*sv_len);
		SGVector<float64_t>::vector_multiply(temp1.vector, outzsv.vector, Ysv.vector, sv_len);
		// in case sv_len < x_n tempg != dot(temp1, Xsv);
		float64_t tempg = 0.0;
		for (auto i = 0; i < sv_len; ++i)
			tempg += temp1[i]*Xsv[i];
		g=wd+(t*dd);
		g-=tempg;

		// Calculation of second derivative 'h'
		SGVector<float64_t> tmp_Xsv = SGVector<float64_t>(Xsv, sv_len, false);
		h = linalg::dot(tmp_Xsv, tmp_Xsv);
		
		h+=dd;

		// Calculation of 1D Newton step 'd'
		t-=g/h;
		if (((g*g)/h)<1e-10)
			break;

	} while(1);

	sg_memcpy(out, outz.vector, sizeof(float64_t)*x_n);
}

void CNewtonSVM::obj_fun_linear()
{
	SGVector<float64_t> weights = get_w();
	SGVector<float64_t> v = binary_labels(m_labels)->get_labels();

	for (int32_t i=0; i<x_n; i++)
	{
		if (out[i]<0)
			out[i]=0;
	}

	//create copy of w0
	SGVector<float64_t> w0(x_d+1);
	sg_memcpy(w0, weights, sizeof(float64_t)*(x_d));
	w0[x_d]=0; //do not penalize b

	//create copy of out
	float64_t* out1=SG_MALLOC(float64_t, x_n);

	//compute steps for obj
	SGVector<float64_t>::vector_multiply(out1, out, out, x_n);
	float64_t p1=SGVector<float64_t>::sum(out1, x_n)/2;
	float64_t C1;
	
	SGVector<float64_t> w0copy(x_d+1);
	sg_memcpy(w0copy.vector, w0.vector, sizeof(float64_t)*(x_d+1));
	w0copy.scale_vector(0.5, w0copy, x_d+1);
	C1 = lambda*linalg::dot(w0, w0copy);
	obj=p1+C1;
	SGVector<float64_t>::scale_vector(lambda, w0, x_d);
	float64_t* temp=SG_CALLOC(float64_t, x_n); //temp = out.*Y
	SGVector<float64_t>::vector_multiply(temp, out, v.vector, x_n);
	float64_t* temp1=SG_CALLOC(float64_t, x_d);
	SGVector<float64_t> vec;

	for (int32_t i=0; i<x_n; i++)
	{
		features->add_to_dense_vec(temp[i], i, temp1, x_d);
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

	size_sv=sv_len;

	SG_FREE(out1);
	SG_FREE(temp);
	SG_FREE(temp1);
	SG_FREE(p2);
}
#endif //HAVE_LAPACK
