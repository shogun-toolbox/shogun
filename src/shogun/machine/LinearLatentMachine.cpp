/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Thoralf Klein, Sergey Lisitsyn, Soeren Sonnenburg
 */

#include <shogun/base/progress.h>
#include <shogun/features/LatentFeatures.h>
#include <shogun/latent/LatentModel.h>
#include <shogun/machine/LinearLatentMachine.h>

using namespace shogun;

LinearLatentMachine::LinearLatentMachine()
	: LinearMachine()
{
	init();
}

LinearLatentMachine::LinearLatentMachine(std::shared_ptr<LatentModel> model, float64_t C)
	: LinearMachine()
{
	init();
	m_C= C;
	set_model(model);

	index_t feat_dim = m_model->get_dim();
	SGVector<float64_t> w;
	w.resize_vector(feat_dim);
	w.zero();
	set_w(w);
}

LinearLatentMachine::~LinearLatentMachine()
{

}

std::shared_ptr<LatentLabels> LinearLatentMachine::apply_latent(std::shared_ptr<Features> data)
{
	if (m_model == NULL)
		SG_ERROR("LatentModel is not set!\n")

	auto lf = std::dynamic_pointer_cast<LatentFeatures>(data);
	m_model->set_features(lf);

	return apply_latent();
}

void LinearLatentMachine::set_model(std::shared_ptr<LatentModel> latent_model)
{
	ASSERT(latent_model != NULL)


	m_model = latent_model;
}

bool LinearLatentMachine::train_machine(std::shared_ptr<Features> data)
{
	if (m_model == NULL)
		SG_ERROR("LatentModel is not set!\n")

	SG_DEBUG("PSI size: %d\n", m_model->get_dim())
	SG_DEBUG("Number of training data: %d\n", m_model->get_num_vectors())
	SG_DEBUG("Initialise PSI (x,h)\n")
	m_model->cache_psi_features();

	/*
	 * define variables for calculating the stopping
	 * criterion for the outer loop
	 */
	float64_t decrement = 0.0, primal_obj = 0.0, prev_po = 0.0;
	float64_t inner_eps = 0.5*m_C*m_epsilon;
	bool stop = false;
	m_cur_iter = 0;
	auto pb = SG_PROGRESS(range(m_max_iter));
	/* do CCCP */
	SG_DEBUG("Starting CCCP\n")
	while ((m_cur_iter < 2)||(!stop&&(m_cur_iter < m_max_iter)))
	{
		SG_DEBUG("iteration: %d\n", m_cur_iter)
		/* do the SVM optimisation with fixed h* */
		SG_DEBUG("Do the inner loop of CCCP: optimize for w for fixed h*\n")
		primal_obj = do_inner_loop(inner_eps);

		/* calculate the decrement */
		decrement = prev_po - primal_obj;
		prev_po = primal_obj;
		SG_DEBUG("decrement: %f\n", decrement)
		SG_DEBUG("primal objective: %f\n", primal_obj)

		/* check the stopping criterion */
		stop = (inner_eps < (0.5*m_C*m_epsilon+1E-8)) && (decrement < m_C*m_epsilon);

		inner_eps = -decrement*0.01;
		inner_eps = Math::max(inner_eps, 0.5*m_C*m_epsilon);
		SG_DEBUG("inner epsilon: %f\n", inner_eps)

		/* find argmaxH */
		SG_DEBUG("Find and set h_i = argmax_h (w, psi(x_i,h))\n")
		m_model->argmax_h(get_w());

		SG_DEBUG("Recalculating PSI (x,h) with the new h variables\n")
		m_model->cache_psi_features();

		/* increment iteration counter */
		m_cur_iter++;
		pb.print_progress();
	}
	pb.complete();
	return true;
}

void LinearLatentMachine::init()
{
	m_C = 10.0;
	m_epsilon = 1E-3;
	m_max_iter = 400;
	m_model = NULL;

	SG_ADD(&m_C, "C", "Cost constant.");
	SG_ADD(&m_epsilon, "epsilon", "Convergence precision.");
	SG_ADD(&m_max_iter, "max_iter", "Maximum iterations.");
	SG_ADD(&m_model, "latent_model", "Latent Model.");
}

