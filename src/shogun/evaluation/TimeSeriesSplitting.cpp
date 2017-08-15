/*
 * Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2017 Sahil Chaddha
 */

#include <shogun/evaluation/TimeSeriesSplitting.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

CTimeSeriesSplitting::CTimeSeriesSplitting() : CSplittingStrategy()
{
	m_rng = sg_rand;
}

CTimeSeriesSplitting::CTimeSeriesSplitting(CLabels* labels, index_t num_subsets)
    : CSplittingStrategy(labels, num_subsets)
{
	m_rng = sg_rand;
}

void CTimeSeriesSplitting::build_subsets()
{
	reset_subsets();
	m_is_filled = true;

	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	index_t num_subsets = m_subset_indices->get_num_elements();

	for (auto i = 0; i < num_subsets; ++i)
	{
		CDynamicArray<index_t>* current =
		    (CDynamicArray<index_t>*)m_subset_indices->get_element(i);

		/* filling current with indices on right end  */
		for (auto k = i == num_subsets - 1
		                  ? indices.vlen - m_h
		                  : (i + 1) * (indices.vlen / num_subsets);
		     k < indices.vlen; ++k)
		{
			current->append_element(indices.vector[k]);
		}

		/* unref */
		SG_UNREF(current);
	}

	m_subset_indices->shuffle(m_rng);
}

/* To ensure to get h value in future in test set*/
void CTimeSeriesSplitting::set_h(index_t h)
{
	index_t num_subsets = m_subset_indices->get_num_elements();
	index_t num_labels = m_labels->get_num_labels();

	/* h value should not be greater than difference between number of labels
	 * and start index of second last split point */
	if (h >= num_labels - (num_subsets - 1) * (num_labels / num_subsets))
	{
		SG_WARNING("h can not be %d. Setting h value to default 1.\n", h);
		m_h = 1;
		return;
	}
	m_h = h;
}

index_t CTimeSeriesSplitting::get_h()
{
	return m_h;
}