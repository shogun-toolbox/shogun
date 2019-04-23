/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Sanuj Sharma, Bjoern Esser
 */

#include <limits>
#include <queue>
#include <algorithm>
#include <functional>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/View.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/RelaxedTree.h>
#include <shogun/multiclass/tree/RelaxedTreeUtil.h>

using namespace shogun;

RelaxedTree::RelaxedTree()
	:m_max_num_iter(3), m_A(0.5), m_B(5), m_svm_C(1), m_svm_epsilon(0.001),
	m_kernel(NULL), m_feats(NULL), m_machine_for_confusion_matrix(NULL), m_num_classes(0)
{
	SG_ADD(&m_max_num_iter, "m_max_num_iter", "max number of iterations in alternating optimization");
	SG_ADD(&m_svm_C, "m_svm_C", "C for svm", ParameterProperties::HYPER);
	SG_ADD(&m_A, "m_A", "parameter A", ParameterProperties::HYPER);
	SG_ADD(&m_B, "m_B", "parameter B", ParameterProperties::HYPER);
	SG_ADD(&m_svm_epsilon, "m_svm_epsilon", "epsilon for svm", ParameterProperties::HYPER);
}

RelaxedTree::~RelaxedTree()
{



}

std::shared_ptr<MulticlassLabels> RelaxedTree::apply_multiclass(std::shared_ptr<Features> data)
{
	if (data != NULL)
	{
		auto feats = data->as<DenseFeatures<float64_t>>();
		set_features(feats);
	}

	// init kernels for all sub-machines
	for (auto m: m_machines)
	{
		auto machine = m->as<SVM>();
		auto kernel = machine->get_kernel();
		auto lhs = kernel->get_lhs();
		kernel->init(lhs, m_feats);
	}

	auto lab = std::make_shared<MulticlassLabels>(m_feats->get_num_vectors());

	for (int32_t i=0; i < lab->get_num_labels(); ++i)
	{
		lab->set_int_label(i, int32_t(apply_one(i)));
	}

	return lab;
}

float64_t RelaxedTree::apply_one(int32_t idx)
{
	auto node = m_root->as<bnode_t>();
	int32_t klass = -1;
	while (node != NULL)
	{
		auto svm = m_machines.at(node->machine())->as<SVM>();
		float64_t result = svm->apply_one(idx);

		if (result < 0)
		{
			// go left
			if (node->left()) // has left subtree
			{
				node = node->left();
			}
			else // stop here
			{
				for (int32_t i=0; i < node->data.mu.vlen; ++i)
				{
					if (node->data.mu[i] <= 0 && node->data.mu[i] > -2)
					{
						klass = i;
						break;
					}
				}
				node = NULL;
			}
		}
		else
		{
			// go right
			if (node->right())
			{
				node = node->right();
			}
			else
			{
				for (int32_t i=0; i <node->data.mu.vlen; ++i)
				{
					if (node->data.mu[i] >= 0)
					{
						klass = i;
						break;
					}
				}
				node = NULL;
			}
		}


	}

	return klass;
}

bool RelaxedTree::train_machine(std::shared_ptr<Features> data)
{
	if (m_machine_for_confusion_matrix == NULL)
		SG_ERROR("Call set_machine_for_confusion_matrix before training\n")
	if (m_kernel == NULL)
		SG_ERROR("assign a valid kernel before training\n")

	if (data)
	{
		set_features(data->template as<DenseFeatures<float64_t>>());
	}

	auto lab = multiclass_labels(m_labels);

	RelaxedTreeUtil util;
	SGMatrix<float64_t> conf_mat = util.estimate_confusion_matrix(
			m_machine_for_confusion_matrix->as<BaseMulticlassMachine>(),
			m_feats, lab, m_num_classes);

	// train root
	SGVector<int32_t> classes(m_num_classes);

	for (int32_t i=0; i < m_num_classes; ++i)
		classes[i] = i;


	m_root = train_node(conf_mat, classes);

	std::queue<std::shared_ptr<bnode_t>> node_q;
	node_q.push(m_root->as<bnode_t>());

	while (node_q.size() != 0)
	{
		auto node = node_q.front();

		// left node
		SGVector <int32_t> left_classes(m_num_classes);
		int32_t k=0;
		for (int32_t i=0; i < m_num_classes; ++i)
		{
			// active classes are labeled as -1 or 0
			// -2 indicate classes that are already pruned away
			if (node->data.mu[i] <= 0 && node->data.mu[i] > -2)
				left_classes[k++] = i;
		}

		left_classes.vlen = k;

		if (left_classes.vlen >= 2)
		{
			auto left_node = train_node(conf_mat, left_classes);
			node->left(left_node);
			node_q.push(left_node);
		}

		// right node
		SGVector <int32_t> right_classes(m_num_classes);
		k=0;
		for (int32_t i=0; i < m_num_classes; ++i)
		{
			// active classes are labeled as 0 or +1
			if (node->data.mu[i] >= 0)
				right_classes[k++] = i;
		}

		right_classes.vlen = k;

		if (right_classes.vlen >= 2)
		{
			auto right_node = train_node(conf_mat, right_classes);
			node->right(right_node);
			node_q.push(right_node);
		}

		node_q.pop();
	}

	return true;
}

std::shared_ptr<RelaxedTree::bnode_t> RelaxedTree::train_node(const SGMatrix<float64_t> &conf_mat, SGVector<int32_t> classes)
{
	SGVector<int32_t> best_mu;
	std::shared_ptr<SVM> best_svm = NULL;
	float64_t best_score = std::numeric_limits<float64_t>::max();

	std::vector<RelaxedTree::entry_t> mu_init = init_node(conf_mat, classes);
	for (std::vector<RelaxedTree::entry_t>::const_iterator it = mu_init.begin(); it != mu_init.end(); ++it)
	{
		auto svm = std::make_shared<LibSVM>();

		svm->set_store_model_features(true);

		SGVector<int32_t> mu = train_node_with_initialization(*it, classes, svm);
		float64_t score = compute_score(mu, svm);

		if (score < best_score)
		{
			best_score = score;
			best_mu = mu;

			best_svm = svm;
		}
	}

	auto node = std::make_shared<bnode_t>();


	m_machines.push_back(best_svm);
	node->machine(m_machines.size()-1);

	SGVector<int32_t> long_mu(m_num_classes);
	std::fill(&long_mu[0], &long_mu[m_num_classes], -2);
	for (int32_t i=0; i < best_mu.vlen; ++i)
	{
		if (best_mu[i] == 1)
			long_mu[classes[i]] = 1;
		else if (best_mu[i] == -1)
			long_mu[classes[i]] = -1;
		else if (best_mu[i] == 0)
			long_mu[classes[i]] = 0;
	}

	node->data.mu = long_mu;
	return node;
}

float64_t RelaxedTree::compute_score(SGVector<int32_t> mu, std::shared_ptr<SVM >svm)
{
	float64_t num_pos=0, num_neg=0;
	for (int32_t i=0; i < mu.vlen; ++i)
	{
		if (mu[i] == 1)
			num_pos++;
		else if (mu[i] == -1)
			num_neg++;
	}

	int32_t totalSV = svm->get_support_vectors().vlen;
	float64_t score = num_neg/(num_neg+num_pos) * totalSV/num_pos +
		num_pos/(num_neg+num_pos)*totalSV/num_neg;
	return score;
}

SGVector<int32_t> RelaxedTree::train_node_with_initialization(const RelaxedTree::entry_t &mu_entry, SGVector<int32_t> classes, std::shared_ptr<SVM >svm)
{
	SGVector<int32_t> mu(classes.vlen), prev_mu(classes.vlen);
	mu.zero();
	mu[mu_entry.first.first] = 1;
	mu[mu_entry.first.second] = -1;

	SGVector<int32_t> long_mu(m_num_classes);
	svm->set_C(m_svm_C, m_svm_C);
	svm->set_epsilon(m_svm_epsilon);

	auto labs = multiclass_labels(m_labels);
	for (int32_t iiter=0; iiter < m_max_num_iter; ++iiter)
	{
		long_mu.zero();
		for (int32_t i=0; i < classes.vlen; ++i)
		{
			if (mu[i] == 1)
				long_mu[classes[i]] = 1;
			else if (mu[i] == -1)
				long_mu[classes[i]] = -1;
		}

		SGVector<int32_t> subset(m_feats->get_num_vectors());
		SGVector<float64_t> binlab(m_feats->get_num_vectors());
		int32_t k=0;

		for (int32_t i=0; i < binlab.vlen; ++i)
		{
			int32_t lab = labs->get_int_label(i);
			binlab[i] = long_mu[lab];
			if (long_mu[lab] != 0)
				subset[k++] = i;
		}

		subset.vlen = k;

		auto binary_labels = std::make_shared<BinaryLabels>(binlab);
		auto feats_train = view(m_feats, subset);
		auto labels_train = view(binary_labels, subset);

		auto kernel = m_kernel->shallow_copy()->as<Kernel>();
		kernel->init(feats_train, feats_train);
		svm->set_kernel(kernel);
		svm->set_labels(labels_train);
		svm->train();

		std::copy(&mu[0], &mu[mu.vlen], &prev_mu[0]);

		mu = color_label_space(svm, classes);

		bool bbreak = true;
		for (int32_t i=0; i < mu.vlen; ++i)
		{
			if (mu[i] != prev_mu[i])
			{
				bbreak = false;
				break;
			}
		}

		if (bbreak)
			break;
	}

	return mu;
}

struct EntryComparator
{
	bool operator() (const RelaxedTree::entry_t& e1, const RelaxedTree::entry_t& e2)
	{
		return e1.second < e2.second;
	}
};
std::vector<RelaxedTree::entry_t> RelaxedTree::init_node(const SGMatrix<float64_t> &global_conf_mat, SGVector<int32_t> classes)
{
	// local confusion matrix
	SGMatrix<float64_t> conf_mat(classes.vlen, classes.vlen);
	for (index_t i=0; i < conf_mat.num_rows; ++i)
	{
		for (index_t j=0; j < conf_mat.num_cols; ++j)
		{
			conf_mat(i, j) = global_conf_mat(classes[i], classes[j]);
		}
	}

	// make conf matrix symmetry
	for (index_t i=0; i < conf_mat.num_rows; ++i)
	{
		for (index_t j=0; j < conf_mat.num_cols; ++j)
		{
			conf_mat(i,j) += conf_mat(j,i);
		}
	}

	std::vector<RelaxedTree::entry_t> entries;
	for (index_t i=0; i < classes.vlen; ++i)
	{
		for (index_t j=i+1; j < classes.vlen; ++j)
		{
			entries.push_back(std::make_pair(std::make_pair(i, j), conf_mat(i,j)));
		}
	}

	std::sort(entries.begin(), entries.end(), EntryComparator());

	const size_t max_n_samples = 30;
	int32_t n_samples = std::min(max_n_samples, entries.size());

	return std::vector<RelaxedTree::entry_t>(entries.begin(), entries.begin() + n_samples);
}

SGVector<int32_t> RelaxedTree::color_label_space(std::shared_ptr<SVM >svm, SGVector<int32_t> classes)
{
	SGVector<int32_t> mu(classes.vlen);
	auto labels = multiclass_labels(m_labels);

	SGVector<float64_t> resp = eval_binary_model_K(svm);
	ASSERT(resp.vlen == labels->get_num_labels())

	SGVector<float64_t> xi_pos_class(classes.vlen), xi_neg_class(classes.vlen);
	SGVector<float64_t> delta_pos(classes.vlen), delta_neg(classes.vlen);

	for (int32_t i=0; i < classes.vlen; ++i)
	{
		// find number of instances from this class
		int32_t ni=0;
		for (int32_t j=0; j < labels->get_num_labels(); ++j)
		{
			if (labels->get_int_label(j) == classes[i])
			{
				ni++;
			}
		}

		xi_pos_class[i] = 0;
		xi_neg_class[i] = 0;
		for (int32_t j=0; j < resp.vlen; ++j)
		{
			if (labels->get_int_label(j) == classes[i])
			{
				xi_pos_class[i] += std::max(0.0, 1 - resp[j]);
				xi_neg_class[i] += std::max(0.0, 1 + resp[j]);
			}
		}

		delta_pos[i] = 1.0/ni * xi_pos_class[i] - float64_t(m_A)/m_svm_C;
		delta_neg[i] = 1.0/ni * xi_neg_class[i] - float64_t(m_A)/m_svm_C;

		if (delta_pos[i] > 0 && delta_neg[i] > 0)
		{
			mu[i] = 0;
		}
		else
		{
			if (delta_pos[i] < delta_neg[i])
				mu[i] = 1;
			else
				mu[i] = -1;
		}

	}

	// enforce balance constraints
	int32_t B_prime = 0;
	for (int32_t i=0; i < mu.vlen; ++i)
		B_prime += mu[i];

	if (B_prime > m_B)
	{
		enforce_balance_constraints_upper(mu, delta_neg, delta_pos, B_prime, xi_neg_class);
	}
	if (B_prime < -m_B)
	{
		enforce_balance_constraints_lower(mu, delta_neg, delta_pos, B_prime, xi_neg_class);
	}

	int32_t npos = 0;
	for (index_t i=0; i < mu.vlen; ++i)
	{
		if (mu[i] == 1)
			npos++;
	}

	if (npos == 0)
	{
		// no positive class
		index_t min_idx = Math::arg_min(xi_pos_class.vector, 1, xi_pos_class.vlen);
		mu[min_idx] = 1;
	}

	int32_t nneg = 0;
	for (index_t i=0; i < mu.vlen; ++i)
	{
		if (mu[i] == -1)
			nneg++;
	}

	if (nneg == 0)
	{
		// no negative class
		index_t min_idx = Math::arg_min(xi_neg_class.vector, 1, xi_neg_class.vlen);
		if (mu[min_idx] == 1 && (npos == 0 || npos == 1))
		{
			// avoid overwritting the only positive class
			float64_t min_val = 0;
			int32_t i, min_i;
			for (i=0; i < xi_neg_class.vlen; ++i)
			{
				if (mu[i] != 1)
				{
					min_val = xi_neg_class[i];
					break;
				}
			}
			min_i = i;
			for (i=i+1; i < xi_neg_class.vlen; ++i)
			{
				if (mu[i] != 1 && xi_neg_class[i] < min_val)
				{
					min_val = xi_neg_class[i];
					min_i = i;
				}
			}
			mu[min_i] = -1;
		}
		else
		{
			mu[min_idx] = -1;
		}
	}

	return mu;
}

void RelaxedTree::enforce_balance_constraints_upper(SGVector<int32_t> &mu, SGVector<float64_t> &delta_neg,
		SGVector<float64_t> &delta_pos, int32_t B_prime, SGVector<float64_t>& xi_neg_class)
{
	SGVector<index_t> index_zero = mu.find(0);
	SGVector<index_t> index_pos = mu.find_if(std::bind(std::less<int32_t>(), 0, std::placeholders::_1));

	int32_t num_zero = index_zero.vlen;
	int32_t num_pos  = index_pos.vlen;

	SGVector<index_t> class_index(num_zero+2*num_pos);
	std::copy(&index_zero[0], &index_zero[num_zero], &class_index[0]);
	std::copy(&index_pos[0], &index_pos[num_pos], &class_index[num_zero]);
	std::copy(&index_pos[0], &index_pos[num_pos], &class_index[num_pos+num_zero]);

	SGVector<int32_t> orig_mu(num_zero + 2*num_pos);
	orig_mu.zero();
	std::fill(&orig_mu[num_zero], &orig_mu[orig_mu.vlen], 1);

	SGVector<int32_t> delta_steps(num_zero+2*num_pos);
	std::fill(&delta_steps[0], &delta_steps[delta_steps.vlen], 1);

	SGVector<int32_t> new_mu(num_zero + 2*num_pos);
	new_mu.zero();
	std::fill(&new_mu[0], &new_mu[num_zero], -1);

	SGVector<float64_t> S_delta(num_zero + 2*num_pos);
	S_delta.zero();
	for (index_t i=0; i < num_zero; ++i)
		S_delta[i] = delta_neg[index_zero[i]];

	for (int32_t i=0; i < num_pos; ++i)
	{
		float64_t delta_k = delta_neg[index_pos[i]];
		float64_t delta_k_0 = -delta_pos[index_pos[i]];

		index_t tmp_index = num_zero + i*2;
		if (delta_k_0 <= delta_k)
		{
			new_mu[tmp_index] = 0;
			new_mu[tmp_index+1] = -1;

			S_delta[tmp_index] = delta_k_0;
			S_delta[tmp_index+1] = delta_k;

			delta_steps[tmp_index] = 1;
			delta_steps[tmp_index+1] = 1;
		}
		else
		{
			new_mu[tmp_index] = -1;
			new_mu[tmp_index+1] = 0;

			S_delta[tmp_index] = (delta_k_0+delta_k)/2;
			S_delta[tmp_index+1] = delta_k_0;

			delta_steps[tmp_index] = 2;
			delta_steps[tmp_index+1] = 1;
		}
	}

	SGVector<index_t> sorted_index = Math::argsort(S_delta);
	SGVector<float64_t> S_delta_sorted(S_delta.vlen);
	for (index_t i=0; i < sorted_index.vlen; ++i)
	{
		S_delta_sorted[i] = S_delta[sorted_index[i]];
		new_mu[i] = new_mu[sorted_index[i]];
		orig_mu[i] = orig_mu[sorted_index[i]];
		class_index[i] = class_index[sorted_index[i]];
		delta_steps[i] = delta_steps[sorted_index[i]];
	}

	SGVector<int32_t> valid_flag(S_delta.vlen);
	std::fill(&valid_flag[0], &valid_flag[valid_flag.vlen], 1);

	int32_t d=0;
	int32_t ctr=0;

	while (true)
	{
		if (d == B_prime - m_B || d == B_prime - m_B + 1)
			break;

		while (!valid_flag[ctr])
			ctr++;

		if (delta_steps[ctr] == 1)
		{
			mu[class_index[ctr]] = new_mu[ctr];
			d++;
		}
		else
		{
			// this case should happen only when rho >= 1
			if (d <= B_prime - m_B - 2)
			{
				mu[class_index[ctr]] = new_mu[ctr];
				ASSERT(new_mu[ctr] == -1)
				d += 2;
				for (index_t i=0; i < class_index.vlen; ++i)
				{
					if (class_index[i] == class_index[ctr])
						valid_flag[i] = 0;
				}
			}
			else
			{
				float64_t Delta_k_minus = 2*S_delta_sorted[ctr];

				// find the next smallest Delta_j or Delta_{j,0}
				float64_t Delta_j_min=0;
				int32_t j=0;
				for (int32_t itr=ctr+1; itr < S_delta_sorted.vlen; ++itr)
				{
					if (valid_flag[itr] == 0)
						continue;

					if (delta_steps[itr] == 1)
					{
						j=itr;
						Delta_j_min = S_delta_sorted[j];
					}
				}

				// find the largest Delta_i or Delta_{i,0}
				float64_t Delta_i_max = 0;
				int32_t i=-1;
				for (int32_t itr=ctr-1; itr >= 0; --itr)
				{
					if (delta_steps[itr] == 1 && valid_flag[itr] == 1)
					{
						i = itr;
						Delta_i_max = S_delta_sorted[i];
					}
				}

				// find the l with the largest Delta_l_minus - Delta_l_0
				float64_t Delta_l_max = std::numeric_limits<float64_t>::min();
				int32_t l=-1;
				for (int32_t itr=ctr-1; itr >= 0; itr--)
				{
					if (delta_steps[itr] == 2)
					{
						float64_t delta_tmp = xi_neg_class[class_index[itr]];
						if (delta_tmp > Delta_l_max)
						{
							l = itr;
							Delta_l_max = delta_tmp;
						}
					}
				}

				// one-step-min = j
				if (Delta_j_min <= Delta_k_minus - Delta_i_max &&
						Delta_j_min <= Delta_k_minus - Delta_l_max)
				{
					mu[class_index[j]] = new_mu[j];
					d++;
				}
				else
				{
					// one-step-min = Delta_k_minus - Delta_i_max
					if (Delta_k_minus - Delta_i_max <= Delta_j_min &&
							Delta_k_minus - Delta_i_max <= Delta_k_minus - Delta_l_max)
					{
						mu[class_index[ctr]] = -1;
						if (i > 0)
						{
							mu[class_index[i]] = orig_mu[i];
							d++;
						}
						else
						{
							d += 2;
						}
					}
					else
					{
						ASSERT(l > 0)
						mu[class_index[l]] = 0;
						mu[class_index[ctr]] = -1;
						d++;
					}
				}

			}
		}
	}
}

void RelaxedTree::enforce_balance_constraints_lower(SGVector<int32_t> &mu, SGVector<float64_t> &delta_neg,
		SGVector<float64_t> &delta_pos, int32_t B_prime, SGVector<float64_t>& xi_neg_class)
{
	SGVector<index_t> index_zero = mu.find(0);
	SGVector<index_t> index_neg = mu.find_if(std::bind(std::greater<int32_t>(), 0, std::placeholders::_1));

	int32_t num_zero = index_zero.vlen;
	int32_t num_neg  = index_neg.vlen;

	SGVector<index_t> class_index(num_zero+2*num_neg);
	std::copy(&index_zero[0], &index_zero[num_zero], &class_index[0]);
	std::copy(&index_neg[0], &index_neg[num_neg], &class_index[num_zero]);
	std::copy(&index_neg[0], &index_neg[num_neg], &class_index[num_neg+num_zero]);

	SGVector<int32_t> orig_mu(num_zero + 2*num_neg);
	orig_mu.zero();
	std::fill(&orig_mu[num_zero], &orig_mu[orig_mu.vlen], -1);

	SGVector<int32_t> delta_steps(num_zero+2*num_neg);
	std::fill(&delta_steps[0], &delta_steps[delta_steps.vlen], 1);

	SGVector<int32_t> new_mu(num_zero + 2*num_neg);
	new_mu.zero();
	std::fill(&new_mu[0], &new_mu[num_zero], 1);

	SGVector<float64_t> S_delta(num_zero + 2*num_neg);
	S_delta.zero();
	for (index_t i=0; i < num_zero; ++i)
		S_delta[i] = delta_pos[index_zero[i]];

	for (int32_t i=0; i < num_neg; ++i)
	{
		float64_t delta_k = delta_pos[index_neg[i]];
		float64_t delta_k_0 = -delta_neg[index_neg[i]];

		index_t tmp_index = num_zero + i*2;
		if (delta_k_0 <= delta_k)
		{
			new_mu[tmp_index] = 0;
			new_mu[tmp_index+1] = 1;

			S_delta[tmp_index] = delta_k_0;
			S_delta[tmp_index+1] = delta_k;

			delta_steps[tmp_index] = 1;
			delta_steps[tmp_index+1] = 1;
		}
		else
		{
			new_mu[tmp_index] = 1;
			new_mu[tmp_index+1] = 0;

			S_delta[tmp_index] = (delta_k_0+delta_k)/2;
			S_delta[tmp_index+1] = delta_k_0;

			delta_steps[tmp_index] = 2;
			delta_steps[tmp_index+1] = 1;
		}
	}

	SGVector<index_t> sorted_index = Math::argsort(S_delta);
	SGVector<float64_t> S_delta_sorted(S_delta.vlen);
	for (index_t i=0; i < sorted_index.vlen; ++i)
	{
		S_delta_sorted[i] = S_delta[sorted_index[i]];
		new_mu[i] = new_mu[sorted_index[i]];
		orig_mu[i] = orig_mu[sorted_index[i]];
		class_index[i] = class_index[sorted_index[i]];
		delta_steps[i] = delta_steps[sorted_index[i]];
	}

	SGVector<int32_t> valid_flag(S_delta.vlen);
	std::fill(&valid_flag[0], &valid_flag[valid_flag.vlen], 1);

	int32_t d=0;
	int32_t ctr=0;

	while (true)
	{
		if (d == -m_B - B_prime || d == -m_B - B_prime + 1)
			break;

		while (!valid_flag[ctr])
			ctr++;

		if (delta_steps[ctr] == 1)
		{
			mu[class_index[ctr]] = new_mu[ctr];
			d++;
		}
		else
		{
			// this case should happen only when rho >= 1
			if (d >= -m_B - B_prime - 2)
			{
				mu[class_index[ctr]] = new_mu[ctr];
				ASSERT(new_mu[ctr] == 1)
				d += 2;

				for (index_t i=0; i < class_index.vlen; ++i)
				{
					if (class_index[i] == class_index[ctr])
						valid_flag[i] = 0;
				}
			}
			else
			{
				float64_t Delta_k_minus = 2*S_delta_sorted[ctr];

				// find the next smallest Delta_j or Delta_{j,0}
				float64_t Delta_j_min=0;
				int32_t j=0;
				for (int32_t itr=ctr+1; itr < S_delta_sorted.vlen; ++itr)
				{
					if (valid_flag[itr] == 0)
						continue;

					if (delta_steps[itr] == 1)
					{
						j=itr;
						Delta_j_min = S_delta_sorted[j];
					}
				}

				// find the largest Delta_i or Delta_{i,0}
				float64_t Delta_i_max = 0;
				int32_t i=-1;
				for (int32_t itr=ctr-1; itr >= 0; --itr)
				{
					if (delta_steps[itr] == 1 && valid_flag[itr] == 1)
					{
						i = itr;
						Delta_i_max = S_delta_sorted[i];
					}
				}

				// find the l with the largest Delta_l_minus - Delta_l_0
				float64_t Delta_l_max = std::numeric_limits<float64_t>::min();
				int32_t l=-1;
				for (int32_t itr=ctr-1; itr >= 0; itr--)
				{
					if (delta_steps[itr] == 2)
					{
						float64_t delta_tmp = xi_neg_class[class_index[itr]];
						if (delta_tmp > Delta_l_max)
						{
							l = itr;
							Delta_l_max = delta_tmp;
						}
					}
				}

				// one-step-min = j
				if (Delta_j_min <= Delta_k_minus - Delta_i_max &&
						Delta_j_min <= Delta_k_minus - Delta_l_max)
				{
					mu[class_index[j]] = new_mu[j];
					d++;
				}
				else
				{
					// one-step-min = Delta_k_minus - Delta_i_max
					if (Delta_k_minus - Delta_i_max <= Delta_j_min &&
							Delta_k_minus - Delta_i_max <= Delta_k_minus - Delta_l_max)
					{
						mu[class_index[ctr]] = -1;
						if (i > 0)
						{
							mu[class_index[i]] = orig_mu[i];
							d++;
						}
						else
						{
							d += 2;
						}
					}
					else
					{
						ASSERT(l > 0)
						mu[class_index[l]] = 0;
						mu[class_index[ctr]] = -1;
						d++;
					}
				}

			}
		}
	}
}

SGVector<float64_t> RelaxedTree::eval_binary_model_K(std::shared_ptr<SVM >svm)
{
	auto lab = svm->apply_regression(m_feats);
	SGVector<float64_t> resp(lab->get_num_labels());
	for (int32_t i=0; i < resp.vlen; ++i)
		resp[i] = lab->get_label(i) - m_A/m_svm_C;

	return resp;
}
