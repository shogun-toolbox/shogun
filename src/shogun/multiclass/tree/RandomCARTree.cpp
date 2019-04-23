/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/multiclass/tree/RandomCARTree.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

RandomCARTree::RandomCARTree()
: CARTree()
{
	init();
}

RandomCARTree::~RandomCARTree()
{
}

void RandomCARTree::set_feature_subset_size(index_t size)
{
	REQUIRE(size>0, "Subset size should be greater than 0. %d supplied!\n",size)
	m_randsubset_size=size;
}

index_t RandomCARTree::compute_best_attribute(const SGMatrix<float64_t>& mat, const SGVector<float64_t>& weights, std::shared_ptr<DenseLabels> labels,
	SGVector<float64_t>& left, SGVector<float64_t>& right, SGVector<bool>& is_left_final, index_t &num_missing_final, index_t &count_left,
	index_t &count_right, index_t subset_size, const SGVector<index_t>& active_indices)

{
	auto num_feats = (m_pre_sort) ? mat.num_cols : mat.num_rows;

	// if subset size is not set choose sqrt(num_feats) by default
	if (m_randsubset_size==0)
		m_randsubset_size = std::sqrt((float64_t)num_feats);
	subset_size=m_randsubset_size;

	REQUIRE(subset_size<=num_feats, "The Feature subset size(set %d) should be less than"
	" or equal to the total number of features(%d here).\n",subset_size,num_feats)

	return CARTree::compute_best_attribute(mat,weights,labels,left,right,is_left_final,num_missing_final,count_left,count_right,subset_size, active_indices);

}

void RandomCARTree::init()
{
	m_randsubset_size=0;

	SG_ADD(&m_randsubset_size,"m_randsubset_size", "random features subset size");
}
