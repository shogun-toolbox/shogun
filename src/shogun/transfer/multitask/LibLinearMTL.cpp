/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Christian Widmer, Sergey Lisitsyn,
 *          Soeren Sonnenburg, Bjoern Esser
 */

#include <utility>
#include <vector>

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/base/progress.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/optimization/liblinear/tron.h>
#include <shogun/transfer/multitask/LibLinearMTL.h>
#include <shogun/mathematics/UniformIntDistribution.h>

using namespace shogun;


	LibLinearMTL::LibLinearMTL()
: RandomMixin<LinearMachine>()
{
	init();
}

LibLinearMTL::LibLinearMTL(
		float64_t C, std::shared_ptr<DotFeatures> traindat, std::shared_ptr<Labels> trainlab)
: RandomMixin<LinearMachine>()
{
	init();
	C1=C;
	C2=C;
	use_bias=true;

	set_features(std::move(traindat));
	set_labels(std::move(trainlab));

}


void LibLinearMTL::init()
{
	use_bias=false;
	C1=1;
	C2=1;
	set_max_iterations();
	epsilon=1e-5;

	SG_ADD(&C1, "C1", "C Cost constant 1.", ParameterProperties::HYPER);
	SG_ADD(&C2, "C2", "C Cost constant 2.", ParameterProperties::HYPER);
	SG_ADD(&use_bias, "use_bias", "Indicates if bias is used.");
	SG_ADD(&epsilon, "epsilon", "Convergence precision.");
	SG_ADD(&max_iterations, "max_iterations", "Max number of iterations.");

}

LibLinearMTL::~LibLinearMTL()
{
}

bool LibLinearMTL::train_machine(std::shared_ptr<Features> data)
{

	ASSERT(m_labels)

	if (data)
	{
		if (!data->has_property(FP_DOT))
			error("Specified features are not of type DotFeatures");

		set_features(data->as<DotFeatures>());
	}
	ASSERT(features)
	m_labels->ensure_valid();


	int32_t num_train_labels=m_labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	if (num_vec!=num_train_labels)
	{
		error("number of vectors {} does not match "
				"number of training labels {}",
				num_vec, num_train_labels);
	}


	float64_t* training_w = NULL;
	if (use_bias)
		training_w=SG_MALLOC(float64_t, num_feat+1);
	else
		training_w=SG_MALLOC(float64_t, num_feat+0);

	liblinear_problem prob;
	if (use_bias)
	{
		prob.n=num_feat+1;
		memset(training_w, 0, sizeof(float64_t)*(num_feat+1));
	}
	else
	{
		prob.n=num_feat;
		memset(training_w, 0, sizeof(float64_t)*(num_feat+0));
	}
	prob.l=num_vec;
	prob.x=features;
	prob.y=SG_MALLOC(float64_t, prob.l);
	prob.use_bias=use_bias;

	auto bl = binary_labels(m_labels);
	for (int32_t i=0; i<prob.l; i++)
		prob.y[i]=bl->get_label(i);

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob.l;i++)
	{
		if(prob.y[i]==+1)
			pos++;
	}
	neg = prob.l - pos;

	io::info("{} training points {} dims", prob.l, prob.n);
	io::info("{} positives, {} negatives", pos, neg);

	double Cp=C1;
	double Cn=C2;
	solve_l2r_l1l2_svc(&prob, epsilon, Cp, Cn);

	if (use_bias)
		set_bias(training_w[num_feat]);
	else
		set_bias(0);

	SG_FREE(prob.y);

	SGVector<float64_t> w(num_feat);
	for (int32_t i=0; i<num_feat; i++)
		w[i] = training_w[i];
	set_w(w);

	return true;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
//		upper_bound_i = Cp if y_i = 1
//		upper_bound_i = Cn if y_i = -1
//		D_ii = 0
// In L2-SVM case:
//		upper_bound_i = INF
//		D_ii = 1/(2*Cp)	if y_i = 1
//		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)


void LibLinearMTL::solve_l2r_l1l2_svc(const liblinear_problem *prob, double eps, double Cp, double Cn)
{



	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = SG_MALLOC(double, l);
	int *index = SG_MALLOC(int, l);
	//double *alpha = SG_MALLOC(double, l);

	int32_t *y = SG_MALLOC(int32_t, l);
	int active_size = l;
	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = Math::INFTY;
	double PGmin_old = -Math::INFTY;
	double PGmax_new, PGmin_new;

	// matrix W
	V = SGMatrix<float64_t>(w_size,num_tasks);

	// save alpha
	alphas = SGVector<float64_t>(l);


	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {Math::INFTY, 0, Math::INFTY};
	if(true)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	int n = prob->n;

	if (prob->use_bias)
		n--;

	// set V to zero
	for(int32_t k=0; k<w_size*num_tasks; k++)
	{
		V.matrix[k] = 0;
	}

	// init alphas
	for(i=0; i<l; i++)
	{
		alphas[i] = 0;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
		QD[i] = diag[GETI(i)];
		QD[i] += prob->x->dot(i, prob->x,i);
		index[i] = i;
	}

	auto pb = SG_PROGRESS(range(10));
	Time start_time;
	UniformIntDistribution<int> uniform_int_dist;
	while (iter < max_iterations)
	{
		COMPUTATION_CONTROLLERS
		if (m_max_train_time > 0 && start_time.cur_time_diff() > m_max_train_time)
			break;

		PGmax_new = -Math::INFTY;
		PGmin_new = Math::INFTY;

		for (i=0; i<active_size; i++)
		{
			int j = uniform_int_dist(m_prng, {i, active_size-1});
			Math::swap(index[i], index[j]);
		}

		for (s=0;s<active_size;s++)
		{
			i = index[s];
			int32_t yi = y[i];
			int32_t ti = task_indicator_lhs[i];
			C = upper_bound[GETI(i)];

			// we compute the inner sum by looping over tasks
			// this update is the main result of MTL_DCD
		    typedef std::map<index_t, float64_t>::const_iterator map_iter;

			float64_t inner_sum = 0;
			for (map_iter it=task_similarity_matrix.data[ti].begin(); it!=task_similarity_matrix.data[ti].end(); it++)
			{

				// get data from sparse matrix
				int32_t e_i = it->first;
                float64_t sim = it->second;

				// fetch vector
				SGVector<float64_t> tmp_w = V.get_column(e_i);
				inner_sum += sim * yi * prob->x->dot(i, tmp_w);

				//possibly deal with bias
				//if (prob->use_bias)
				//	G+=w[n];
			}

			// compute gradient
			G = inner_sum-1.0;

			// check if point can be removed from active set
			PG = 0;
			if (alphas[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					Math::swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alphas[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					Math::swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = Math::max(PGmax_new, PG);
			PGmin_new = Math::min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				// save previous alpha
				double alpha_old = alphas[i];

				// project onto feasible set
				alphas[i] = Math::min(Math::max(alphas[i] - G/QD[i], 0.0), C);
				d = (alphas[i] - alpha_old)*yi;

				// update corresponding weight vector
				float64_t* tmp_w = V.get_column_vector(ti);
				prob->x->add_to_dense_vec(d, i, tmp_w, n);


				//if (prob->use_bias)
				//	w[n]+=d;
			}
		}

		iter++;
		float64_t gap=PGmax_new - PGmin_new;
		pb.print_absolute(
		    gap, -Math::log10(gap), -Math::log10(1), -Math::log10(eps));

		if(gap <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				PGmax_old = Math::INFTY;
				PGmin_old = -Math::INFTY;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = Math::INFTY;
		if (PGmin_old >= 0)
			PGmin_old = -Math::INFTY;
	}

	pb.complete_absolute();
	io::info("optimization finished, #iter = {}",iter);
	if (iter >= max_iterations)
	{
		io::warn("reaching max number of iterations\nUsing -s 2 may be faster"
				"(also see liblinear FAQ)");
	}



	delete [] QD;
	//delete [] alpha;
	delete [] y;
	delete [] index;
}


float64_t LibLinearMTL::compute_primal_obj()
{
	/* python protype
	   num_param = param.shape[0]
	   num_dim = len(all_xt[0])
	   num_tasks = int(num_param / num_dim)
	   num_examples = len(all_xt)

# vector to matrix
W = param.reshape(num_tasks, num_dim)

obj = 0

reg_obj = 0
loss_obj = 0

assert len(all_xt) == len(all_xt) == len(task_indicator)

# L2 regularizer
for t in xrange(num_tasks):
reg_obj += 0.5 * np.dot(W[t,:], W[t,:])

# MTL regularizer
for s in xrange(num_tasks):
for t in xrange(num_tasks):
reg_obj += 0.5 * L[s,t] * np.dot(W[s,:], W[t,:])

# loss
for i in xrange(num_examples):
ti = task_indicator[i]
t = all_lt[i] * np.dot(W[ti,:], all_xt[i])
# hinge
loss_obj += max(0, 1 - t)


# combine to final objective
obj = reg_obj + C * loss_obj


return obj
*/

	io::info("DONE to compute Primal OBJ");
	// calculate objective value
	SGMatrix<float64_t> W = get_W();

	float64_t obj = 0;
	int32_t num_vec = features->get_num_vectors();
	int32_t w_size = features->get_dim_feature_space();

	// L2 regularizer
	for (int32_t t=0; t<num_tasks; t++)
	{
		float64_t* w_t = W.get_column_vector(t);

		for(int32_t i=0; i<w_size; i++)
		{
			obj += 0.5 * w_t[i]*w_t[i];
		}
	}

	// MTL regularizer
	for (int32_t s=0; s<num_tasks; s++)
	{
		float64_t* w_s = W.get_column_vector(s);
		for (int32_t t=0; t<num_tasks; t++)
		{
			float64_t* w_t = W.get_column_vector(t);
			float64_t l = graph_laplacian.matrix[s*num_tasks+t];

			for(int32_t i=0; i<w_size; i++)
			{
				obj += 0.5 * l * w_s[i]*w_t[i];
			}
		}
	}

	// loss
	auto bl = binary_labels(m_labels);
	for(int32_t i=0; i<num_vec; i++)
	{
		int32_t ti = task_indicator_lhs[i];
		SGVector<float64_t> w_t = W.get_column(ti);
		float64_t residual = bl->get_label(i) * features->dot(i, w_t);

		// hinge loss
		obj += C1 * Math::max(0.0, 1 - residual);

	}

	io::info("DONE to compute Primal OBJ, obj={}",obj);

	return obj;
}

float64_t LibLinearMTL::compute_dual_obj()
{
	/* python prototype
	   num_xt = len(xt)

# compute quadratic term
for i in xrange(num_xt):
for j in xrange(num_xt):

s = task_indicator[i]
t = task_indicator[j]

obj -= 0.5 * M[s,t] * alphas[i] * alphas[j] * lt[i] * lt[j] * np.dot(xt[i], xt[j])

return obj
*/

	io::info("starting to compute DUAL OBJ");

	int32_t num_vec=features->get_num_vectors();

	float64_t obj = 0;

	// compute linear term
	for(int32_t i=0; i<num_vec; i++)
	{
		obj += alphas[i];
	}

	// compute quadratic term

	int32_t v_size = features->get_dim_feature_space();

	// efficient computation
	for (int32_t s=0; s<num_tasks; s++)
	{
		float64_t* v_s = V.get_column_vector(s);
		for (int32_t t=0; t<num_tasks; t++)
		{
			float64_t* v_t = V.get_column_vector(t);
			const float64_t ts = task_similarity_matrix(s, t);

			for(int32_t i=0; i<v_size; i++)
			{
				obj -= 0.5 * ts * v_s[i]*v_t[i];
			}
		}
	}

	/*
	// naiive implementation
	float64_t tmp_val2 = 0;

	for(int32_t i=0; i<num_vec; i++)
	{
		int32_t ti_i = task_indicator_lhs[i];
		for(int32_t j=0; j<num_vec; j++)
		{
			// look up task similarity
			int32_t ti_j = task_indicator_lhs[j];

			const float64_t ts = task_similarity_matrix(ti_i, ti_j);

			// compute objective
			tmp_val2 -= 0.5 * alphas[i] * alphas[j] * ts * ((BinaryLabels*)m_labels)->get_label(i) *
				((BinaryLabels*)m_labels)->get_label(j) * features->dot(i, features,j);
		}
	}
	*/


	return obj;
}


float64_t LibLinearMTL::compute_duality_gap()
{
	return 0.0;
}


#endif //HAVE_LAPACK
