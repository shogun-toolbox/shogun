/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Pan Deng, Bjoern Esser,
 *          Sanuj Sharma
 */

#include <algorithm>

#include <shogun/base/progress.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/Time.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/ShareBoost.h>
#include <shogun/multiclass/ShareBoostOptimizer.h>

using namespace shogun;

ShareBoost::ShareBoost()
	:LinearMulticlassMachine(), m_nonzero_feas(0)
{
	init_sb_params();
}

ShareBoost::ShareBoost(const std::shared_ptr<DenseFeatures<float64_t> >&features, const std::shared_ptr<MulticlassLabels >&labs, int32_t num_nonzero_feas)
	:LinearMulticlassMachine(std::make_shared<MulticlassOneVsRestStrategy>(), features, NULL, labs), m_nonzero_feas(num_nonzero_feas)
{
	init_sb_params();
}

void ShareBoost::init_sb_params()
{
	SG_ADD(&m_nonzero_feas, "nonzero_feas", "Number of non-zero features");
	SG_ADD(&m_activeset, "active_set", "Selected features");
}

SGVector<int32_t> ShareBoost::get_activeset()
{
	return m_activeset;
}

bool ShareBoost::train_machine(std::shared_ptr<Features> data)
{
	if (data)
		set_features(data);
	auto fea = m_features->as<DenseFeatures<float64_t>>();

	if (m_features == NULL)
		error("No features given for training");
	if (m_labels == NULL)
		error("No labels given for training");

	init_strategy();

	if (m_nonzero_feas <= 0)
		error("Set a valid (> 0) number of non-zero features to seek before training");
	if (m_nonzero_feas > fea->get_num_features())
		error("Number of non-zero features ({}) cannot be larger than number of features ({}) in the data",
				m_nonzero_feas, fea->get_num_features());

	m_fea = fea->get_feature_matrix();
	m_rho = SGMatrix<float64_t>(m_multiclass_strategy->get_num_classes(), m_fea.num_cols);
	m_rho_norm = SGVector<float64_t>(m_fea.num_cols);
	m_pred = SGMatrix<float64_t>(m_fea.num_cols, m_multiclass_strategy->get_num_classes());
	m_pred.zero();

	m_activeset = SGVector<int32_t>(m_fea.num_rows);
	m_activeset.vlen = 0;

	m_machines.clear();
	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
		m_machines.push_back(std::make_shared<LinearMachine>());

	auto timer = std::make_shared<Time>();

	float64_t t_compute_pred = 0; // t of 1st round is 0, since no pred to compute
	for (auto t : SG_PROGRESS(range(m_nonzero_feas)))
	{
		timer->start();
		compute_rho();
		int32_t i_fea = choose_feature();
		m_activeset.vector[m_activeset.vlen] = i_fea;
		m_activeset.vlen += 1;
		float64_t t_choose_feature = timer->cur_time_diff();
		timer->start();
		optimize_coefficients();
		float64_t t_optimize = timer->cur_time_diff();

		SG_DEBUG(" SB[round {:03d}]: ({:8.4f} + {:8.4f}) sec.", t,
				t_compute_pred + t_choose_feature, t_optimize);

		timer->start();
		compute_pred();
		t_compute_pred = timer->cur_time_diff();
	}



	// release memory
	m_fea = SGMatrix<float64_t>();
	m_rho = SGMatrix<float64_t>();
	m_rho_norm = SGVector<float64_t>();
	m_pred = SGMatrix<float64_t>();

	return true;
}

void ShareBoost::compute_pred()
{
	auto fea = m_features->as<DenseFeatures<float64_t>>();
	auto subset_fea = std::make_shared<DenseSubsetFeatures<float64_t>>(fea, m_activeset);
	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
	{
		auto machine = m_machines.at(i)->as<LinearMachine>();
		auto lab = machine->apply_regression(subset_fea);
		SGVector<float64_t> lab_raw = lab->get_labels();
		std::copy(lab_raw.vector, lab_raw.vector + lab_raw.vlen, m_pred.get_column_vector(i));


	}
}

void ShareBoost::compute_pred(const float64_t *W)
{
	int32_t w_len = m_activeset.vlen;

	for (int32_t i=0; i < m_multiclass_strategy->get_num_classes(); ++i)
	{
		auto machine = m_machines.at(i)->as<LinearMachine>();
		SGVector<float64_t> w(w_len);
		std::copy(W + i*w_len, W + (i+1)*w_len, w.vector);
		machine->set_w(w);

	}
	compute_pred();
}

void ShareBoost::compute_rho()
{
	auto lab = multiclass_labels(m_labels);

	for (int32_t i=0; i < m_rho.num_rows; ++i)
	{ // i loop classes
		for (int32_t j=0; j < m_rho.num_cols; ++j)
		{ // j loop samples
			int32_t label = lab->get_int_label(j);

			m_rho(i, j) =
			    std::exp((label == i) - m_pred(j, label) + m_pred(j, i));
		}
	}

	// normalize
	for (int32_t j=0; j < m_rho.num_cols; ++j)
	{
		m_rho_norm[j] = 0;
		for (int32_t i=0; i < m_rho.num_rows; ++i)
			m_rho_norm[j] += m_rho(i,j);
	}
}

int32_t ShareBoost::choose_feature()
{
	SGVector<float64_t> l1norm(m_fea.num_rows);
	auto lab = multiclass_labels(m_labels);
	for (int32_t j=0; j < m_fea.num_rows; ++j)
	{
		if (std::find(&m_activeset[0], &m_activeset[m_activeset.vlen], j) !=
				&m_activeset[m_activeset.vlen])
		{
			l1norm[j] = 0;
		}
		else
		{
			l1norm[j] = 0;
			for (int32_t k=0; k < m_multiclass_strategy->get_num_classes(); ++k)
			{
				float64_t abssum = 0;
				for (int32_t ii=0; ii < m_fea.num_cols; ++ii)
				{
					abssum += m_fea(j, ii)*(m_rho(k, ii)/m_rho_norm[ii] -
							(j == lab->get_int_label(ii)));
				}
				l1norm[j] += Math::abs(abssum);
			}
			l1norm[j] /= m_fea.num_cols;
		}
	}

	return Math::arg_max(l1norm.vector, 1, l1norm.vlen);
}

void ShareBoost::optimize_coefficients()
{
	ShareBoostOptimizer optimizer(shared_from_this()->as<ShareBoost>(), false);
	optimizer.optimize();
}

void ShareBoost::set_features(const std::shared_ptr<Features >&f)
{
	auto fea = f->as<DenseFeatures<float64_t>>();
	if (fea == NULL)
		error("Require DenseFeatures<float64_t>");
	LinearMulticlassMachine::set_features(fea);
}
