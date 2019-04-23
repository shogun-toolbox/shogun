/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang, Shell Hu, Sergey Lisitsyn,
 *          Bjoern Esser, Sanuj Sharma
 */

#include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

MulticlassOneVsOneStrategy::MulticlassOneVsOneStrategy()
	:MulticlassStrategy(), m_num_machines(0), m_num_samples(SGVector<int32_t>())
{
	register_parameters();
}

MulticlassOneVsOneStrategy::MulticlassOneVsOneStrategy(EProbHeuristicType prob_heuris)
	:MulticlassStrategy(prob_heuris), m_num_machines(0), m_num_samples(SGVector<int32_t>())
{
	register_parameters();
}

void MulticlassOneVsOneStrategy::register_parameters()
{
	//SG_ADD(&m_num_samples, "num_samples", "Number of samples in each training machine");
	SG_WARNING("%s::MulticlassOneVsOneStrategy(): register parameters!\n", get_name());
}

void MulticlassOneVsOneStrategy::train_start(std::shared_ptr<MulticlassLabels >orig_labels, std::shared_ptr<BinaryLabels >train_labels)
{
	MulticlassStrategy::train_start(orig_labels, train_labels);
	m_num_machines=m_num_classes*(m_num_classes-1)/2;

	m_train_pair_idx_1 = 0;
	m_train_pair_idx_2 = 1;

	m_num_samples.resize_vector(m_num_machines);
}

bool MulticlassOneVsOneStrategy::train_has_more()
{
	return m_train_iter < m_num_machines;
}

SGVector<int32_t> MulticlassOneVsOneStrategy::train_prepare_next()
{
	MulticlassStrategy::train_prepare_next();

	SGVector<int32_t> subset(m_orig_labels->get_num_labels());
	int32_t tot=0;
	auto mc_orig = multiclass_labels(m_orig_labels);
	auto binary_train = binary_labels(m_train_labels);
	for (int32_t k=0; k < m_orig_labels->get_num_labels(); ++k)
	{
		if (mc_orig->get_int_label(k)==m_train_pair_idx_1)
		{
			binary_train->set_label(k, +1.0);
			subset[tot]=k;
			tot++;
		}
		else if (mc_orig->get_int_label(k)==m_train_pair_idx_2)
		{
			binary_train->set_label(k, -1.0);
			subset[tot]=k;
			tot++;
		}
	}

	m_train_pair_idx_2++;
	if (m_train_pair_idx_2 >= m_num_classes)
	{
		m_train_pair_idx_1++;
		m_train_pair_idx_2=m_train_pair_idx_1+1;
	}

	// collect num samples each machine
	m_num_samples[m_train_iter-1] = tot;

	subset.resize_vector(tot);
	return subset;
}

int32_t MulticlassOneVsOneStrategy::decide_label(SGVector<float64_t> outputs)
{
	// if OVO with prob outputs, find max posterior
	if (outputs.vlen==m_num_classes)
		return Math::arg_max(outputs.vector, 1, outputs.vlen);

	int32_t s=0;
	SGVector<int32_t> votes(m_num_classes);
    SGVector<int32_t> dec_vals(m_num_classes);
	votes.zero();
    dec_vals.zero();

	for (int32_t i=0; i<m_num_classes; i++)
	{
		for (int32_t j=i+1; j<m_num_classes; j++)
		{
			if (outputs[s]>0)
            {
				votes[i]++;
                dec_vals[i] += Math::abs(outputs[s]);
            }
			else
            {
				votes[j]++;
                dec_vals[j] += Math::abs(outputs[s]);
            }
            s++;
		}
	}

    int32_t i_max=0;
    int32_t vote_max=-1;
    float64_t dec_val_max=-1;

    for (int32_t i=0; i < m_num_classes; ++i)
    {
        if (votes[i] > vote_max)
        {
            i_max = i;
            vote_max = votes[i];
            dec_val_max = dec_vals[i];
        }
        else if (votes[i] == vote_max)
        {
            if (dec_vals[i] > dec_val_max)
            {
                i_max = i;
                dec_val_max = dec_vals[i];
            }
        }
    }

    return i_max;
}

void MulticlassOneVsOneStrategy::rescale_outputs(SGVector<float64_t> outputs)
{
	if (m_num_machines < 1)
		return;

	SGVector<int32_t> indx1(m_num_machines);
	SGVector<int32_t> indx2(m_num_machines);

	int32_t tot = 0;
	for (int32_t j=0; j<m_num_classes; j++)
	{
		for (int32_t k=j+1; k<m_num_classes; k++)
		{
			indx1[tot] = j;
			indx2[tot] = k;
			tot++;
		}
	}

	if(tot!=m_num_machines)
		SG_ERROR("%s::rescale_output(): size(outputs) is not num_machines.\n", get_name());

	switch(get_prob_heuris_type())
	{
		case OVO_PRICE:
			rescale_heuris_price(outputs,indx1,indx2);
			break;
		case OVO_HASTIE:
			rescale_heuris_hastie(outputs,indx1,indx2);
			break;
		case OVO_HAMAMURA:
			rescale_heuris_hamamura(outputs,indx1,indx2);
			break;
		case PROB_HEURIS_NONE:
			break;
		default:
			SG_ERROR("%s::rescale_outputs(): Unknown OVO probability heuristic type!\n", get_name());
			break;
	}
}

void MulticlassOneVsOneStrategy::rescale_heuris_price(SGVector<float64_t> outputs,
		const SGVector<int32_t> indx1, const SGVector<int32_t> indx2)
{
	if (m_num_machines != outputs.vlen)
	{
		SG_ERROR("%s::rescale_heuris_price(): size(outputs) = %d != m_num_machines = %d\n",
				get_name(), outputs.vlen, m_num_machines);
	}

	SGVector<float64_t> new_outputs(m_num_classes);
	new_outputs.zero();

	for (int32_t j=0; j<m_num_classes; j++)
	{
		for (int32_t m=0; m<m_num_machines; m++)
		{
			if (indx1[m]==j)
				new_outputs[j] += 1.0 / (outputs[m]+1E-12);
			if (indx2[m]==j)
				new_outputs[j] += 1.0 / (1.0-outputs[m]+1E-12);
		}

		new_outputs[j] = 1.0 / (new_outputs[j] - m_num_classes + 2);
	}

	//outputs.resize_vector(m_num_classes);

	float64_t norm = SGVector<float64_t>::sum(new_outputs);
	for (int32_t i=0; i<new_outputs.vlen; i++)
		outputs[i] = new_outputs[i] / norm;
}

void MulticlassOneVsOneStrategy::rescale_heuris_hastie(SGVector<float64_t> outputs,
		const SGVector<int32_t> indx1, const SGVector<int32_t> indx2)
{
	if (m_num_machines != outputs.vlen)
	{
		SG_ERROR("%s::rescale_heuris_hastie(): size(outputs) = %d != m_num_machines = %d\n",
				get_name(), outputs.vlen, m_num_machines);
	}

	SGVector<float64_t> new_outputs(m_num_classes);
	new_outputs.zero();

	for (int32_t j=0; j<m_num_classes; j++)
	{
		for (int32_t m=0; m<m_num_machines; m++)
		{
			if (indx1[m]==j)
				new_outputs[j] += outputs[m];
			if (indx2[m]==j)
				new_outputs[j] += 1.0-outputs[m];
		}

		new_outputs[j] *= 2.0 / (m_num_classes * (m_num_classes - 1));
		new_outputs[j] += 1E-10;
	}

	SGVector<float64_t> mu(m_num_machines);
	SGVector<float64_t> prev_outputs(m_num_classes);
	float64_t gap = 1.0;

	while (gap > 1E-12)
	{
		prev_outputs = new_outputs.clone();

		for (int32_t m=0; m<m_num_machines; m++)
			mu[m] = new_outputs[indx1[m]] / (new_outputs[indx1[m]] + new_outputs[indx2[m]]);

		for (int32_t j=0; j<m_num_classes; j++)
		{
			float64_t numerator = 0.0;
			float64_t denominator = 0.0;
			for (int32_t m=0; m<m_num_machines; m++)
			{
				if (indx1[m]==j)
				{
					numerator += m_num_samples[m] * outputs[m];
					denominator += m_num_samples[m] * mu[m];
				}

				if (indx2[m]==j)
				{
					numerator += m_num_samples[m] * (1.0-outputs[m]);
					denominator += m_num_samples[m] * (1.0-mu[m]);
				}
			}

			// update posterior
			new_outputs[j] *= numerator / denominator;
		}

		float64_t norm = SGVector<float64_t>::sum(new_outputs);
		for (int32_t i=0; i<new_outputs.vlen; i++)
			new_outputs[i] /= norm;

		// gap is Euclidean distance
		for (int32_t i=0; i<new_outputs.vlen; i++)
			prev_outputs[i] -= new_outputs[i];

		gap = SGVector<float64_t>::qsq(prev_outputs.vector, prev_outputs.vlen, 2);
		SG_DEBUG("[Hastie's heuristic] gap = %.12f\n", gap);
	}

	for (int32_t i=0; i<new_outputs.vlen; i++)
		outputs[i] = new_outputs[i];
}

void MulticlassOneVsOneStrategy::rescale_heuris_hamamura(SGVector<float64_t> outputs,
		const SGVector<int32_t> indx1, const SGVector<int32_t> indx2)
{
	if (m_num_machines != outputs.vlen)
	{
		SG_ERROR("%s::rescale_heuris_hamamura(): size(outputs) = %d != m_num_machines = %d\n",
				get_name(), outputs.vlen, m_num_machines);
	}

	SGVector<float64_t> new_outputs(m_num_classes);
	SGVector<float64_t>::fill_vector(new_outputs.vector, new_outputs.vlen, 1.0);

	for (int32_t j=0; j<m_num_classes; j++)
	{
		for (int32_t m=0; m<m_num_machines; m++)
		{
			if (indx1[m]==j)
				new_outputs[j] *= outputs[m];
			if (indx2[m]==j)
				new_outputs[j] *= 1-outputs[m];
		}

		new_outputs[j] += 1E-10;
	}

	float64_t norm = SGVector<float64_t>::sum(new_outputs);

	for (int32_t i=0; i<new_outputs.vlen; i++)
		outputs[i] = new_outputs[i] / norm;
}

