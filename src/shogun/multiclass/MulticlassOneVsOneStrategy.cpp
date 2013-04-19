/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

CMulticlassOneVsOneStrategy::CMulticlassOneVsOneStrategy()
	:CMulticlassStrategy(), m_num_machines(0)
{
}

void CMulticlassOneVsOneStrategy::train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels)
{
	CMulticlassStrategy::train_start(orig_labels, train_labels);
	m_num_machines=m_num_classes*(m_num_classes-1)/2;

	m_train_pair_idx_1 = 0;
	m_train_pair_idx_2 = 1;
}

bool CMulticlassOneVsOneStrategy::train_has_more()
{
	return m_train_iter < m_num_machines;
}

SGVector<int32_t> CMulticlassOneVsOneStrategy::train_prepare_next()
{
	CMulticlassStrategy::train_prepare_next();

	SGVector<int32_t> subset(m_orig_labels->get_num_labels());
	int32_t tot=0;
	for (int32_t k=0; k < m_orig_labels->get_num_labels(); ++k)
	{
		if (((CMulticlassLabels*) m_orig_labels)->get_int_label(k)==m_train_pair_idx_1)
		{
			((CBinaryLabels*) m_train_labels)->set_label(k, +1.0);
			subset[tot]=k;
			tot++;
		}
		else if (((CMulticlassLabels*) m_orig_labels)->get_int_label(k)==m_train_pair_idx_2)
		{
			((CBinaryLabels*) m_train_labels)->set_label(k, -1.0);
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

	subset.resize_vector(tot);
	return subset;
}

int32_t CMulticlassOneVsOneStrategy::decide_label(SGVector<float64_t> outputs)
{
    if (outputs.vlen==m_num_classes)
    {
	    return SGVector<float64_t>::arg_max(outputs.vector, 1, outputs.vlen);
    }
    // if length of outputs is not c(c-1)/2 or c
    if (outputs.vlen!=m_num_machines)
    {
        SG_ERROR("Dimension of outputs are incorrect");
    }

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
                dec_vals[i] += CMath::abs(outputs[s]);
            }
			else
            {
				votes[j]++;
                dec_vals[j] += CMath::abs(outputs[s]);
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

/** OVO method 3 in Jonathan Milgram
 * "One Against One" or "One Against All"
 * Which One is Better for Handwriting Recognition with SVMs?
 */
SGVector<float64_t> CMulticlassOneVsOneStrategy::rescale_output(SGVector<float64_t> outputs)
{
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

    ASSERT(tot==m_num_machines);

    SGVector<float64_t> posterior(m_num_classes);
    SGVector<float64_t>::fill_vector(posterior.vector, posterior.vlen, 1.0);
    for (int32_t j=0; j<m_num_classes; j++)
    {
        for (int32_t m=0; m<m_num_machines; m++)
        {
            if (indx1[m]==j)
                posterior[j] *= outputs[m]; 
            if (indx2[m]==j)
                posterior[j] *= 1-outputs[m]; 
        }
    }

    float64_t norm = SGVector<float64_t>::sum(posterior);
    for (int32_t i=0; i<posterior.vlen; i++) 
        posterior[i] /= norm;
    return posterior;
}
