
/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Soeren Sonnenburg, Sergey Lisitsyn, Abhinav Rai,
 *          Bjoern Esser
 */

#include <shogun/base/progress.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/optimization/liblinear/tron.h>
#include <shogun/regression/svr/LibLinearRegression.h>

#include <utility>

using namespace shogun;

LibLinearRegression::LibLinearRegression() :
	RandomMixin<LinearMachine>()
{
	register_parameters();
	init_defaults();
}

LibLinearRegression::LibLinearRegression(float64_t C, std::shared_ptr<DotFeatures> feats, std::shared_ptr<Labels> labs) :
	RandomMixin<LinearMachine>()
{
	register_parameters();
	init_defaults();
	set_C(C);
	set_features(std::move(feats));
	set_labels(std::move(labs));
}

void LibLinearRegression::init_defaults()
{
	set_C(1.0);
	set_epsilon(1e-2);
	set_tube_epsilon(1e-1);
	set_max_iter(10000);
	set_use_bias(false);
	set_liblinear_regression_type(L2R_L1LOSS_SVR_DUAL);
}

void LibLinearRegression::register_parameters()
{
	SG_ADD(&m_C, "C", "regularization constant", ParameterProperties::HYPER);
	SG_ADD(
	    &m_tube_epsilon, "tube_epsilon", "svr tube epsilon",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_epsilon, "epsilon", "tolerance of termination criterion",
	    ParameterProperties::SETTING);
	SG_ADD(&m_max_iter, "max_iterations", "max number of iterations",
			ParameterProperties::SETTING);
	SG_ADD(&m_use_bias, "use_bias", "indicates whether bias should be used",
			ParameterProperties::SETTING);
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_liblinear_regression_type,
	    "liblinear_regression_type", "Type of LibLinear regression.",
	    ParameterProperties::SETTING,
	    SG_OPTIONS(L2R_L2LOSS_SVR, L2R_L1LOSS_SVR_DUAL, L2R_L2LOSS_SVR_DUAL));
}

LibLinearRegression::~LibLinearRegression()
{
}

bool LibLinearRegression::train_machine(std::shared_ptr<Features> data)
{

	if (data)
		set_features(data->as<DotFeatures>());

	ASSERT(features)
	ASSERT(m_labels && m_labels->get_label_type()==LT_REGRESSION)

	auto num_train_labels=m_labels->get_num_labels();
	auto num_feat=features->get_dim_feature_space();
	auto num_vec=features->get_num_vectors();

	if (num_vec!=num_train_labels)
	{
		error("number of vectors {} does not match "
				"number of training labels {}",
				num_vec, num_train_labels);
	}

	SGVector<float64_t> w;
	auto prob = liblinear_problem();
	prob.use_bias = get_use_bias();

	if (prob.use_bias)
		w=SGVector<float64_t>(SG_MALLOC(float64_t, num_feat+1), num_feat);
	else
		w=SGVector<float64_t>(num_feat);

	if (prob.use_bias)
	{
		prob.n=w.vlen+1;
		memset(w.vector, 0, sizeof(float64_t)*(w.vlen+1));
	}
	else
	{
		prob.n=w.vlen;
		memset(w.vector, 0, sizeof(float64_t)*(w.vlen+0));
	}
	prob.l=num_vec;
	prob.x=features;
	auto labels = regression_labels(m_labels);

	// store reference to vector locally in order to prevent free-ing
	auto lab = labels->get_labels();
	prob.y = lab.data();

	switch (m_liblinear_regression_type)
	{
		case L2R_L2LOSS_SVR:
		{
			float64_t* Cs = SG_MALLOC(float64_t, prob.l);
			for(int i = 0; i < prob.l; i++)
				Cs[i] = get_C();

			function *fun_obj=new l2r_l2_svr_fun(&prob, Cs, get_tube_epsilon());
			Tron tron_obj(fun_obj, get_epsilon());
			tron_obj.tron(w.vector, m_max_train_time);
			delete fun_obj;
			SG_FREE(Cs);
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(w, &prob);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(w, &prob);
			break;
		default:
			error("Error: unknown regression type");
			break;
	}

	set_w(w);
	if (prob.use_bias)
		set_bias(w.vector[prob.n - 1]);

	return true;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//		upper_bound_i = C
//		lambda_i = 0
// In L2-SVM case:
//		upper_bound_i = INF
//		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

void LibLinearRegression::solve_l2r_l1l2_svr(SGVector<float64_t>& w, const liblinear_problem *prob)
{
	int l = prob->l;
	float64_t C = get_C();
	float64_t p = get_tube_epsilon();
	// number of features, excluding bias
	int w_size;
	if (prob->use_bias)
		w_size = prob->n - 1;
	else
		w_size = prob->n;
	float64_t eps = get_epsilon();
	int i, s, iter = 0;
	int active_size = l;
	int *index = new int[l];

	float64_t d, G, H;
	float64_t Gmax_old = Math::INFTY;
	float64_t Gmax_new, Gnorm1_new;
	float64_t Gnorm1_init = 0.0;
	SGVector<float64_t> beta(l);
	SGVector<float64_t> QD(l);
	float64_t *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	float64_t lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = Math::INFTY;

	if(m_liblinear_regression_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	linalg::zero(beta);
	linalg::zero(w);

	for(i=0; i<l; i++)
	{
		QD[i] = prob->x->dot(i, prob->x,i);
		prob->x->add_to_dense_vec(beta[i], i, w.vector, w_size);

		if (prob->use_bias)
			w.vector[w_size]+=beta[i];

		index[i] = i;
	}

	auto pb = SG_PROGRESS(range(10));
	UniformIntDistribution<int> uniform_int_dist;
	while(iter < m_max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = uniform_int_dist(m_prng, {i, active_size-1});
			Math::swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			G += prob->x->dot(i, w);
			if (prob->use_bias)
				G+=w.vector[w_size];

			float64_t Gp = G+p;
			float64_t Gn = G-p;
			float64_t violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					Math::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					Math::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					Math::swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = Math::max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			float64_t beta_old = beta[i];
			beta[i] = Math::min(Math::max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				prob->x->add_to_dense_vec(d, i, w.vector, w_size);

				if (prob->use_bias)
					w.vector[w_size]+=d;
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;

		pb.print_absolute(
				Gnorm1_new, -Math::log10(Gnorm1_new),
				-Math::log10(eps * Gnorm1_init), -Math::log10(Gnorm1_init));

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				Gmax_old = Math::INFTY;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	pb.complete_absolute();
	io::info("optimization finished, #iter = {}", iter);
	if(iter >= m_max_iter)
		io::info("WARNING: reaching max number of iterations\nUsing -s 11 may be faster");

	// calculate objective value
	int nSV = 0;
	auto v = linalg::dot(w,w)*0.5;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	io::info("Objective value = {}", v);
	io::info("nSV = {}",nSV);

	delete [] index;
}
