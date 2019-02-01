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
	init();
}

CTimeSeriesSplitting::CTimeSeriesSplitting(CLabels* labels, index_t num_subsets)
    : CSplittingStrategy(labels, num_subsets)
{
	init();
}

void CTimeSeriesSplitting::init()
{
	m_rng = sg_rand;
	m_min_subset_size = 1;
	SG_ADD(&m_min_subset_size, "min_subset_size", 
			"The minimum subset size for test set")
}

void CTimeSeriesSplitting::build_subsets()
{
	reset_subsets();
	m_is_filled = true;

	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	index_t num_subsets = m_subset_indices->get_num_elements();
	index_t split_index;

	for (auto i = 0; i < num_subsets; ++i)
	{
		CDynamicArray<index_t>* current =
		    (CDynamicArray<index_t>*)m_subset_indices->get_element(i);

		if (i == num_subsets - 1)
			split_index = indices.vlen - m_min_subset_size;
		else
			split_index = (i + 1) * (indices.vlen / num_subsets);

		/* filling current with indices on right end  */
		for (auto k = split_index; k < indices.vlen; ++k)
		{
			current->append_element(indices.vector[k]);
		}

		SG_UNREF(current);
	}

	m_subset_indices->shuffle(m_rng);
}

void CTimeSeriesSplitting::set_min_subset_size(index_t min_size)
{
	index_t num_subsets = m_subset_indices->get_num_elements();
	index_t num_labels = m_labels->get_num_labels();

	/* min_size should be less than difference between number of labels
	 * and split index of second last split */
	REQUIRE(min_size > 0, "Minimum subset size should be atleast 1.")
	REQUIRE(
	    min_size < num_labels - (num_subsets - 1) * (num_labels / num_subsets),
	    "Minimum subset size can be atmost %d, constrained by number of "
	    "subsets and labels.",
	    num_labels - (num_subsets - 1) * (num_labels / num_subsets) - 1);
	m_min_subset_size = min_size;
}

index_t CTimeSeriesSplitting::get_min_subset_size()
{
	return m_min_subset_size;
}
