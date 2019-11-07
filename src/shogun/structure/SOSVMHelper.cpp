/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu, Bjoern Esser, Thoralf Klein, Viktor Gal, Jiaolong Xu,
 *          Sanuj Sharma
 */

#include <shogun/structure/SOSVMHelper.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

SOSVMHelper::SOSVMHelper() : SGObject()
{
	init();
}

SOSVMHelper::SOSVMHelper(int32_t bufsize) : SGObject()
{
	m_bufsize = bufsize;
	init();
}

SOSVMHelper::~SOSVMHelper()
{
}

void SOSVMHelper::init()
{
	SG_ADD(&m_primal, "primal", "History of primal values");
	SG_ADD(&m_dual, "dual", "History of dual values");
	SG_ADD(&m_duality_gap, "duality_gap", "History of duality gaps");
	SG_ADD(&m_eff_pass, "eff_pass", "Effective passes");
	SG_ADD(&m_train_error, "train_error", "History of training errors");
	SG_ADD(&m_tracker, "tracker", "Tracker of training progress");
	SG_ADD(&m_bufsize, "bufsize", "Buffer size");

	m_tracker = 0;
	m_bufsize = 1000;
	m_primal = SGVector<float64_t>(m_bufsize);
	m_dual = SGVector<float64_t>(m_bufsize);
	m_duality_gap = SGVector<float64_t>(m_bufsize);
	m_eff_pass = SGVector<float64_t>(m_bufsize);
	m_train_error = SGVector<float64_t>(m_bufsize);
	m_primal.zero();
	m_dual.zero();
	m_duality_gap.zero();
	m_eff_pass.zero();
	m_train_error.zero();
}

float64_t SOSVMHelper::primal_objective(SGVector<float64_t> w, const std::shared_ptr<StructuredModel>& model, float64_t lbda)
{
	float64_t hinge_losses = 0.0;
	auto labels = model->get_labels();
	int32_t N = labels->get_num_labels();


	for (int32_t i = 0; i < N; i++)
	{
		// solve the loss-augmented inference for point i
		auto result = model->argmax(w, i);

		// hinge loss for point i
		float64_t hinge_loss_i = result->score;

		if (hinge_loss_i < 0)
			hinge_loss_i = 0;

		hinge_losses += hinge_loss_i;


	}

	return (lbda/2 * linalg::dot(w, w) + hinge_losses/N);
}

float64_t SOSVMHelper::dual_objective(SGVector<float64_t> w, float64_t aloss, float64_t lbda)
{
	return (-lbda/2 * linalg::dot(w, w) + aloss);
}

float64_t SOSVMHelper::average_loss(SGVector<float64_t> w, const std::shared_ptr<StructuredModel>& model, bool is_ub)
{
	float64_t loss = 0.0;
	auto labels = model->get_labels();
	int32_t N = labels->get_num_labels();


	for (int32_t i = 0; i < N; i++)
	{
		// solve the standard inference for point i
		auto result = model->argmax(w, i, is_ub);

		loss += result->delta;


	}

	return loss / N;
}

void SOSVMHelper::add_debug_info(float64_t primal, float64_t eff_pass, float64_t train_error,
		float64_t dual, float64_t dgap)
{
	if (m_tracker >= m_bufsize)
	{
		io::print("{}::add_debug_information(): Buffer overflows! No more values will be recorded!\n",
			get_name());

		return;
	}

	m_primal[m_tracker] = primal;
	m_eff_pass[m_tracker] = eff_pass;
	m_train_error[m_tracker] = train_error;

	if (dgap >= 0)
	{
		m_dual[m_tracker] = dual;
		m_duality_gap[m_tracker] = dgap;
	}

	m_tracker++;
}

SGVector<float64_t> SOSVMHelper::get_primal_values() const
{
	return m_primal;
}

SGVector<float64_t> SOSVMHelper::get_dual_values() const
{
	return m_dual;
}

SGVector<float64_t> SOSVMHelper::get_duality_gaps() const
{
	return m_duality_gap;
}

SGVector<float64_t> SOSVMHelper::get_eff_passes() const
{
	return m_eff_pass;
}

SGVector<float64_t> SOSVMHelper::get_train_errors() const
{
	return m_train_error;
}

void SOSVMHelper::terminate()
{
	m_primal.resize_vector(m_tracker);
	m_dual.resize_vector(m_tracker);
	m_duality_gap.resize_vector(m_tracker);
	m_eff_pass.resize_vector(m_tracker);
	m_train_error.resize_vector(m_tracker);
}
