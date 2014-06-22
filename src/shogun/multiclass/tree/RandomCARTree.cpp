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

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/RandomCARTree.h>

using namespace shogun;

CRandomCARTree::CRandomCARTree()
: CCARTree()
{
	init();
}

CRandomCARTree::~CRandomCARTree()
{
}

void CRandomCARTree::set_feature_subset_size(int32_t size)
{
	REQUIRE(size>0, "Subset size should be greater than 0. %d supplied!\n",size)
	m_randsubset_size=size;
}

int32_t CRandomCARTree::compute_best_attribute(SGMatrix<float64_t> mat, SGVector<float64_t> weights, SGVector<float64_t> labels_vec, 	
	SGVector<float64_t> left, SGVector<float64_t> right, SGVector<bool> is_left_final, int32_t &num_missing_final, int32_t &count_left,
														 int32_t &count_right)
{
	float64_t max_gain=-1;
	int32_t best_attribute=-1;
	int32_t num_vecs=mat.num_cols;
	int32_t num_feats=mat.num_rows;

	REQUIRE(m_randsubset_size<=num_feats, "The Feature subset size(set %d) should be less than"
	" or equal to the total number of features(%d here)\n",m_randsubset_size,num_feats)

	// if subset size is not set choose sqrt(num_feats) by default
	if (m_randsubset_size==0)
		m_randsubset_size=CMath::sqrt(num_feats-0.f);

	// randomly choose w/o replacement the attributes from which best will be chosen
	// randomly permute and choose 1st randsubset_size elements
	SGVector<index_t> idx(num_feats);
	idx.range_fill(0);
	idx.randperm();

	for (int32_t i=0;i<m_randsubset_size;i++)
	{
		// find number of missing data points for chosen attribute
		int32_t num_missing=0;
		for (int32_t j=0;j<num_vecs;j++)
		{
			if (CMath::fequals(mat(idx[i],j),MISSING,0))
				num_missing++;
		}

		// assimilate non-missing features, corresponding weights and labels
		SGVector<float64_t> non_missing_feats(num_vecs-num_missing);
		SGVector<float64_t> non_missing_weights(num_vecs-num_missing);
		SGVector<float64_t> non_missing_labels(num_vecs-num_missing);
		int32_t c=0;
		for (int32_t j=0;j<num_vecs;j++)
		{
			if (!CMath::fequals(mat(idx[i],j),MISSING,0))
			{
				non_missing_feats[c]=mat(idx[i],j);
				non_missing_weights[c]=weights[j];
				non_missing_labels[c++]=labels_vec[j];
			}
		}

		// get unique feature values
		SGVector<float64_t> nm_feats_copy=non_missing_feats.clone();
		int32_t num_unique=nm_feats_copy.unique(nm_feats_copy.vector,nm_feats_copy.vlen);
		// if only one unique value - it cannot be used to split
		if (num_unique==1)
			continue;

		if (m_nominal[idx[i]])
		{
			// test all 2^(I-1)-1 possible division between two nodes
			int32_t num_cases=CMath::pow(2,(num_unique-1));
			for (int32_t k=1;k<num_cases;k++)
			{
				// stores which vectors are assigned to left child
				SGVector<bool> is_left(num_vecs-num_missing);
				// stores which among the categorical values of chosen attribute are assigned left child
				SGVector<bool> feats_left(num_unique);

				// fill feats_left in a unique way corresponding to the case
				for (int32_t p=0;p<num_unique;p++)
					feats_left[p]=((k/CMath::pow(2,p))%(CMath::pow(2,p+1))==1);

				// form is_left
				for (int32_t j=0;j<num_vecs-num_missing;j++)
				{
					// determine categorical value of jth vector
					int32_t index=-1;
					for (int32_t p=0;p<num_unique;p++)
					{
						if (non_missing_feats[j]==nm_feats_copy[p])
						{
							index=p;
							break;
						}
					}

					// assign jth vector a child corresponding to its categorical value
					is_left[j]=feats_left[index];
				}

				float64_t g=gain(non_missing_labels,non_missing_weights,is_left);
				if (g>max_gain)
				{
					best_attribute=idx[i];
					max_gain=g;
					memcpy(is_left_final.vector,is_left.vector,is_left.vlen*sizeof(bool));
					num_missing_final=num_missing;

					count_left=0;
					for (int32_t l=0;l<num_unique;l++)
						count_left=(feats_left[l])?count_left+1:count_left;

					count_right=num_unique-count_left;

					int32_t l=0;
					int32_t r=0;
					for (int32_t w=0;w<num_unique;w++)
					{
						if (feats_left[w])
							left[l++]=nm_feats_copy[w];
						else
							right[r++]=nm_feats_copy[w];
					}
				}
			}
		}
		else
		{
			// find best split for non-nominal attribute - choose threshold (z)  
			for (int32_t j=0;j<num_unique-1;j++)
			{
				// threshold
				float64_t z=nm_feats_copy[j];

				SGVector<bool> is_left(num_vecs-num_missing);
				for (int32_t k=0;k<num_vecs-num_missing;k++)
					is_left[k]=(non_missing_feats[k]<=z);

				float64_t g=gain(non_missing_labels,non_missing_weights,is_left);
				if (g>max_gain)
				{
					max_gain=g;
					best_attribute=idx[i];
					num_missing_final=num_missing;

					count_left=1;
					count_right=1;
					left[0]=z;
					right[0]=z;

					memcpy(is_left_final.vector,is_left.vector,is_left.vlen*sizeof(bool));
				}
			}
		}
	}

	return best_attribute;
}

void CRandomCARTree::init()
{
	m_randsubset_size=0;

	SG_ADD(&m_randsubset_size,"m_randsubset_size", "random features subset size", MS_NOT_AVAILABLE);
}
