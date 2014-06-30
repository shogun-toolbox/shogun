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
	int32_t num_vecs=mat.num_cols;
	int32_t num_feats=mat.num_rows;

	int32_t n_ulabels;
	SGVector<float64_t> ulabels=get_unique_labels(labels_vec,n_ulabels);

	// if all labels same early stop
	if (n_ulabels==1)
		return -1;

	SGVector<float64_t> total_wclasses(n_ulabels);
	total_wclasses.zero();

	SGVector<int32_t> simple_labels(num_vecs);
	float64_t delta=0;
	if (m_mode==PT_REGRESSION)
		delta=m_label_epsilon;

	for (int32_t i=0;i<num_vecs;i++)
	{
		for (int32_t j=0;j<n_ulabels;j++)
		{
			if (CMath::abs(labels_vec[i]-ulabels[j])<=delta)
			{
				simple_labels[i]=j;
				total_wclasses[j]+=weights[i];
				break;
			}
				
		}
	}

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

	float64_t max_gain=MIN_SPLIT_GAIN;
	int32_t best_attribute=-1;
	float64_t best_threshold=0;
	for (int32_t i=0;i<m_randsubset_size;i++)
	{
		SGVector<float64_t> feats(num_vecs);
		for (int32_t j=0;j<num_vecs;j++)
			feats[j]=mat(idx[i],j);

		// O(N*logN)
		SGVector<index_t> sorted_args=feats.argsort();

		// number of non-missing vecs
		int32_t n_nm_vecs=feats.vlen;
		while (feats[sorted_args[n_nm_vecs-1]]==MISSING)
		{
			total_wclasses[simple_labels[sorted_args[n_nm_vecs-1]]]-=weights[sorted_args[n_nm_vecs-1]];
			n_nm_vecs--;
		}

		// if only one unique value - it cannot be used to split
		if (feats[sorted_args[n_nm_vecs-1]]<=feats[sorted_args[0]]+EQ_DELTA)
			continue;

		if (m_nominal[idx[i]])
		{
			SGVector<int32_t> simple_feats(num_vecs);
			simple_feats.fill_vector(simple_feats.vector,simple_feats.vlen,-1);

			// convert to simple values
			simple_feats[sorted_args[0]]=0;
			int32_t c=0;
			for (int32_t j=1;j<n_nm_vecs;j++)
			{
				if (feats[sorted_args[j]]==feats[sorted_args[j-1]])
					simple_feats[sorted_args[j]]=c;
				else
					simple_feats[sorted_args[j]]=(++c);
			}

			SGVector<float64_t> ufeats(c+1);
			ufeats[0]=feats[sorted_args[0]];
			int32_t u=0;
			for (int32_t j=1;j<n_nm_vecs;j++)
			{
				if (feats[sorted_args[j]]==feats[sorted_args[j-1]])
					continue;
				else
					ufeats[++u]=feats[sorted_args[j]];
			}

			// test all 2^(I-1)-1 possible division between two nodes
			int32_t num_cases=CMath::pow(2,c);
			for (int32_t k=1;k<num_cases;k++)
			{
				SGVector<float64_t> wleft(n_ulabels);
				SGVector<float64_t> wright(n_ulabels);
				wleft.zero();
				wright.zero();

				// stores which vectors are assigned to left child
				SGVector<bool> is_left(num_vecs);
				is_left.fill_vector(is_left.vector,is_left.vlen,false);

				// stores which among the categorical values of chosen attribute are assigned left child
				SGVector<bool> feats_left(c+1);

				// fill feats_left in a unique way corresponding to the case
				for (int32_t p=0;p<c+1;p++)
					feats_left[p]=((k/CMath::pow(2,p))%(CMath::pow(2,p+1))==1);

				// form is_left
				for (int32_t j=0;j<n_nm_vecs;j++)
				{
					is_left[sorted_args[j]]=feats_left[simple_feats[sorted_args[j]]];
					if (is_left[sorted_args[j]])
						wleft[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
					else
						wright[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
				}

				float64_t g=0;
				if (m_mode==PT_MULTICLASS)
					g=gain(wleft,wright,total_wclasses);
				else if (m_mode==PT_REGRESSION)
					g=gain(wleft,wright,total_wclasses,ulabels);
				else
					SG_ERROR("Undefined problem statement\n");

				if (g>max_gain)
				{
					best_attribute=idx[i];
					max_gain=g;
					memcpy(is_left_final.vector,is_left.vector,is_left.vlen*sizeof(bool));
					num_missing_final=num_vecs-n_nm_vecs;

					count_left=0;
					for (int32_t l=0;l<c+1;l++)
						count_left=(feats_left[l])?count_left+1:count_left;

					count_right=c+1-count_left;

					int32_t l=0;
					int32_t r=0;
					for (int32_t w=0;w<c+1;w++)
					{
						if (feats_left[w])
							left[l++]=ufeats[w];
						else
							right[r++]=ufeats[w];
					}
				}
			}
		}
		else
		{
			// O(N)
			SGVector<float64_t> right_wclasses=total_wclasses.clone();
			SGVector<float64_t> left_wclasses(n_ulabels);
			left_wclasses.zero();

			// O(N)
			// find best split for non-nominal attribute - choose threshold (z)
			float64_t z=feats[sorted_args[0]];  
			right_wclasses[simple_labels[sorted_args[0]]]-=weights[sorted_args[0]];
			left_wclasses[simple_labels[sorted_args[0]]]+=weights[sorted_args[0]];
			for (int32_t j=1;j<n_nm_vecs;j++)
			{
				if (feats[sorted_args[j]]<=z+EQ_DELTA)
				{	
					right_wclasses[simple_labels[sorted_args[j]]]-=weights[sorted_args[j]];
					left_wclasses[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
					continue;
				}

				// O(F)
				float64_t g=0;
				if (m_mode==PT_MULTICLASS)
					g=gain(left_wclasses,right_wclasses,total_wclasses);
				else if (m_mode==PT_REGRESSION)
					g=gain(left_wclasses,right_wclasses,total_wclasses,ulabels);
				else
					SG_ERROR("Undefined problem statement\n");

				if (g>max_gain)
				{
					max_gain=g;
					best_attribute=idx[i];
					best_threshold=z;
					num_missing_final=num_vecs-n_nm_vecs;
				}

				z=feats[sorted_args[j]];
				if (feats[sorted_args[n_nm_vecs-1]]<=z+EQ_DELTA)
					break;

				right_wclasses[simple_labels[sorted_args[j]]]-=weights[sorted_args[j]];
				left_wclasses[simple_labels[sorted_args[j]]]+=weights[sorted_args[j]];
			}
		}

		// restore total_wclasses
		while (n_nm_vecs<feats.vlen)
		{
			total_wclasses[simple_labels[sorted_args[n_nm_vecs-1]]]+=weights[sorted_args[n_nm_vecs-1]];
			n_nm_vecs++;
		}
	}

	if (best_attribute==-1)
		return -1;

	if (!m_nominal[best_attribute])
	{
		left[0]=best_threshold;
		right[0]=best_threshold;
		count_left=1;
		count_right=1;
		for (int32_t i=0;i<num_vecs;i++)
			is_left_final[i]=(mat(best_attribute,i)<=best_threshold);
	}

	return best_attribute;
}

void CRandomCARTree::init()
{
	m_randsubset_size=0;

	SG_ADD(&m_randsubset_size,"m_randsubset_size", "random features subset size", MS_NOT_AVAILABLE);
}
