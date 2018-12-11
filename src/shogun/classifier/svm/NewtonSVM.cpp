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
	t = 0;
}

CNewtonSVM::CNewtonSVM(
    float64_t c, CDotFeatures* traindat, CLabels* trainlab, int32_t itr)
    : CIterativeMachine<CLinearMachine>()
{
	lambda=1/c;
	num_iter=itr;
	prec=1e-6;
	C=c;
	t = 0;
	set_features(traindat);
	set_labels(trainlab);
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

		SG_REF(data);
		SG_UNREF(m_continue_features);
		m_continue_features = data->as<CDotFeatures>();
	}

	ASSERT(features)

	SGVector<float64_t> train_labels = binary_labels(m_labels)->get_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	//Assigning dimensions for whole class scope
	x_n=num_vec;
	x_d=num_feat;

	ASSERT(num_vec==train_labels.vlen)

	SGVector<float64_t> weights(x_d);
	set_w(weights);
	out = SGVector<float64_t>(x_n);
	out.set_const(1.0);
	weights.set_const(0.0);

	sv = SGVector<int32_t>(x_n);
	sv.set_const(0);
	grad = SGVector<float64_t>(x_d + 1);
	grad.set_const(0.0);
}

void CNewtonSVM::iteration()
{
	obj_fun_linear();
	SGVector<float64_t> weights = get_w();

	SGVector<float64_t> sgv;
	SGMatrix<float64_t> Xsv(x_d, size_sv);
	for (int32_t k = 0; k < size_sv; k++)
	{
		sgv = features->get_computed_dot_feature_vector(sv[k]);
		sg_memcpy(&Xsv.matrix[k * x_d], sgv.data(), sizeof(float64_t) * (x_d));
	}
	Xsv = linalg::transpose_matrix(Xsv);

	SGMatrix<float64_t> lcrossdiag(x_d + 1, x_d + 1);
	SGVector<float64_t> vector(x_d + 1);

	vector.set_const(lambda);

	vector[x_d] = 0;

	SGMatrix<float64_t>::create_diagonal_matrix(
	    lcrossdiag.data(), vector.data(), x_d + 1);
	SGMatrix<float64_t> Xsv2(x_d, x_d);
	linalg::matrix_prod(Xsv, Xsv, Xsv2, true);

	SGVector<float64_t> sum = linalg::colwise_sum(Xsv);

	SGMatrix<float64_t> Xsv2sum(x_d + 1, x_d + 1);

	for (int32_t i = 0; i < x_d; i++)
	{
		for (int32_t j = 0; j < x_d; j++)
			Xsv2sum(i, j) = Xsv2(i, j);

		Xsv2sum(i, x_d) = sum[i];
	}

	for (int32_t j = 0; j < x_d; j++)
		Xsv2sum(x_d, j) = sum[j];

	Xsv2sum(x_d, x_d) = size_sv;

	linalg::add(Xsv2sum, lcrossdiag, Xsv2sum);

	SGVector<float64_t> step(x_d + 1);
	SGVector<float64_t> s2(x_d + 1);
	s2 = linalg::matrix_prod(linalg::pinvh(Xsv2sum), grad);

	for (int32_t i = 0; i < x_d + 1; i++)
		step[i] = -s2[i];

	line_search_linear(step);

	SGVector<float64_t> tmp_step(step.data(), x_d, false);
	linalg::add(weights, tmp_step, weights, 1.0, t);
	bias += t * step[x_d];
	float64_t newton_decrement = -0.5 * linalg::dot(step, grad);

	if (newton_decrement * 2 < prec * obj)
		m_complete = true;
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
		Xd[i] = features->dense_dot(i, d.data(), x_d);

	linalg::add_scalar(Xd, d[x_d]);

	SGVector<float64_t> tmp_d = SGVector<float64_t>(d.data(), x_d, false);
	float64_t wd = lambda * linalg::dot(weights, tmp_d);
	float64_t dd = lambda * linalg::dot(tmp_d, tmp_d);

	float64_t g, h;
	int32_t sv_len=0;
	SGVector<int32_t> sv(x_n);

	while (1)
	{
		temp1 = linalg::element_prod(Y, Xd);
		temp1forout = temp1.clone();
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

		temp1.zero();
		SGVector<float64_t>::vector_multiply(
		    temp1.data(), outzsv.data(), Ysv.data(), sv_len);
		// in case sv_len < x_n tempg != dot(temp1, Xsv);
		float64_t tempg = 0.0;
		for (auto i = 0; i < sv_len; ++i)
			tempg += temp1[i]*Xsv[i];
		g=wd+(t*dd);
		g-=tempg;

		// Calculation of second derivative 'h'
		SGVector<float64_t> tmp_Xsv = SGVector<float64_t>(Xsv, sv_len, false);
		h = linalg::dot(tmp_Xsv, tmp_Xsv) + dd;
		// Calculation of 1D Newton step 'd'
		t-=g/h;
		if (((g*g)/h)<1e-10)
			break;
	}

	out = outz.clone();
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
	SGVector<float64_t> w0(x_d + 1);
	sg_memcpy(w0, weights, sizeof(float64_t)*(x_d));
	w0[x_d]=0; //do not penalize b

	//create copy of out
	SGVector<float64_t> out1(x_n);

	//compute steps for obj
	out1 = linalg::element_prod(out, out);
	float64_t p1 = linalg::sum(out1) / 2;

	SGVector<float64_t> w0copy(x_d + 1);
	w0copy = w0.clone();
	linalg::scale(w0copy, w0copy, 0.5);
	float64_t C1 = lambda * linalg::dot(w0, w0copy);
	obj = p1 + C1;
	linalg::scale(w0, w0, lambda);
	SGVector<float64_t> temp = linalg::element_prod(out, v);
	SGVector<float64_t> temp1(x_d);

	for (int32_t i=0; i<x_n; i++)
	{
		features->add_to_dense_vec(temp[i], i, temp1, x_d);
	}

	SGVector<float64_t> p2(x_d + 1);

	sg_memcpy(p2, temp1, sizeof(float64_t) * (x_d));

	p2[x_d] = linalg::sum(temp);
	linalg::add(w0, p2, grad, 1.0, -1.0);
	int32_t sv_len=0;

	for (int32_t i=0; i<x_n; i++)
	{
		if (out[i]>0)
			sv[sv_len++]=i;
	}

	size_sv = sv_len;
}
#endif //HAVE_LAPACK
