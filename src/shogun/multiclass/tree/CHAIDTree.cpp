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
#include <shogun/mathematics/Statistics.h>
#include <shogun/multiclass/tree/CHAIDTree.h>

using namespace shogun;

const float64_t CCHAIDTree::MISSING=CMath::MAX_REAL_NUMBER;

CCHAIDTree::CCHAIDTree()
: CTreeMachine<CHAIDTreeNodeData>()
{
	init();
}

CCHAIDTree::CCHAIDTree(int32_t dependent_vartype)
: CTreeMachine<CHAIDTreeNodeData>()
{
	init();
	m_dependent_vartype=dependent_vartype;
}

CCHAIDTree::~CCHAIDTree()
{
}

EProblemType CCHAIDTree::get_machine_problem_type() const
{
	switch (m_dependent_vartype)
	{
		case 0:
			return PT_MULTICLASS;
		case 1:
			return PT_MULTICLASS;
		case 2:
			return PT_REGRESSION;
		default:
			SG_ERROR("Invalid dependent variable type set (%d). Problem type undefined\n",m_dependent_vartype);
	}

	return PT_MULTICLASS;
}

bool CCHAIDTree::is_label_valid(CLabels* lab) const
{
	switch (m_dependent_vartype)
	{
		case 0:
			return lab->get_label_type()==LT_MULTICLASS;
		case 1:
			return lab->get_label_type()==LT_MULTICLASS;
		case 2:
			return lab->get_label_type()==LT_REGRESSION;
		default:
			SG_ERROR("Invalid dependent variable type set (%d). Problem type undefined\n",m_dependent_vartype);
	}

	return false;
}

CMulticlassLabels* CCHAIDTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")
	return new CMulticlassLabels(); 
}

CRegressionLabels* CCHAIDTree::apply_regression(CFeatures* data)
{
	REQUIRE(data, "Data required for regression in apply_regression\n")
	return new CRegressionLabels();
}

void CCHAIDTree::set_weights(SGVector<float64_t> w)
{
	m_weights=w;
	m_weights_set=true;
}

SGVector<float64_t> CCHAIDTree::get_weights() const
{
	if (!m_weights_set)
		SG_ERROR("weights not set\n");

	return m_weights;
}

void CCHAIDTree::clear_weights()
{
	m_weights=SGVector<float64_t>();
	m_weights_set=false;
}

void CCHAIDTree::set_feature_types(SGVector<int32_t> ft)
{
	m_feature_types=ft;
}

SGVector<int32_t> CCHAIDTree::get_feature_types() const
{
	return m_feature_types;
}

void CCHAIDTree::clear_feature_types()
{
	m_feature_types=SGVector<int32_t>();
}

void CCHAIDTree::set_dependent_vartype(int32_t var)
{
	REQUIRE(((var==0)||(var==1)||(var==2)), "Expected 0 or 1 or 2 as argument. %d received\n",var)
	m_dependent_vartype=var;
}

bool CCHAIDTree::train_machine(CFeatures* data)
{
	REQUIRE(data, "Data required for training\n")
	REQUIRE(data->get_feature_class()==C_DENSE,"Dense data required for training\n")

	SGMatrix<float64_t> fmat=(CDenseFeatures<float64_t>::obtain_from_generic(data))->get_feature_matrix();

	REQUIRE(m_feature_types.vlen==fmat.num_rows,"Either feature types are not set or number of feature types specified" 
		" (%d here) is not same is number of features in data matrix (%d here)\n",m_feature_types.vlen,fmat.num_rows)

	if (m_weights_set)
	{
		REQUIRE(m_weights.vlen==fmat.num_cols,"Length of weights vector (currently %d) should be same as" 
					" number of vectors in data (presently %d)",m_weights.vlen,fmat.num_cols)
	}
	else
	{
		// all weights are equal to 1
		m_weights=SGVector<float64_t>(fmat.num_cols);
		m_weights.fill_vector(m_weights.vector,m_weights.vlen,1.0);
	}

	// continuous to ordinal conversion - NOTE: data matrix gets updated
	bool updated=continuous_to_ordinal(fmat);

	SGVector<int32_t> feature_types_cache;
	if (updated)
	{
		// change m_feature_types momentarily
		feature_types_cache=m_feature_types.clone();
		for (int32_t i=0;i<m_feature_types.vlen;i++)
		{
			if (m_feature_types[i]==2)
				m_feature_types[i]=1;
		}
	}

	set_root(CHAIDtrain(data,m_weights,m_labels,0));

	// restore feature types
	if (updated)
		m_feature_types=feature_types_cache;

	return true;
}

CTreeMachineNode<CHAIDTreeNodeData>* CCHAIDTree::CHAIDtrain(CFeatures* data, SGVector<float64_t> weights, CLabels* labels, int32_t level)
{
	REQUIRE(data,"data matrix cannot be empty\n");
	REQUIRE(labels,"labels cannot be NULL\n");

	node_t* node=new node_t();
	SGVector<float64_t> labels_vec=(dynamic_cast<CDenseLabels*>(labels))->get_labels();
	SGMatrix<float64_t> mat=(CDenseFeatures<float64_t>::obtain_from_generic(data))->get_feature_matrix();
	int32_t num_feats=mat.num_rows;
	int32_t num_vecs=mat.num_cols;

	// calculate node label
	if (m_dependent_vartype==2)
	{
		// sum_of_squared_deviation
		node->data.weight_minus_node=sum_of_squared_deviation(labels_vec,weights,node->data.node_label);
		node->data.total_weight=weights.sum(weights);
	}
	else if (m_dependent_vartype==0 || m_dependent_vartype==1)
	{
		SGVector<float64_t> lab=labels_vec.clone();	
		lab.qsort();
		// stores max total weight for a single label 
		int32_t max=weights[0];
		// stores one of the indices having max total weight
		int32_t maxi=0;
		int32_t c=weights[0];
		for (int32_t i=1;i<lab.vlen;i++)
		{
			if (lab[i]==lab[i-1])
			{
				c+=weights[i];
			}
			else if (c>max)
			{
				max=c;
				maxi=i-1;
				c=weights[i];
			}
			else
			{
				c=weights[i];
			}
		}

		if (c>max)
		{
			max=c;
			maxi=lab.vlen-1;
		}

		node->data.node_label=lab[maxi];
		node->data.total_weight=weights.sum(weights);
		node->data.weight_minus_node=node->data.total_weight-max;

	}
	else
	{
		SG_ERROR("dependent variable type should be either 0(nominal) or 1(ordinal) or 2(continuous)\n");	
	}

	// check stopping rules
	// case 1 : all labels same
	SGVector<float64_t> lab=labels_vec.clone();
	int32_t unique=lab.unique(lab.vector,lab.vlen);
	if (unique==1)
		return node;

	// case 2 : all non-dependent attributes (not MISSING) are same
	bool flag=true;
	for (int32_t v=1;v<num_vecs;v++)
	{
		for (int32_t f=0;f<num_feats;f++)
		{
			if ((mat(f,v)!=MISSING) && (mat(f,v-1)!=MISSING))
			{
				if (mat(f,v)!=mat(f,v-1))
				{
					flag=false;
					break;
				}
			}
		}

		if (!flag)
			break;
	}

	if (flag)
		return node;

	// case 3 : current tree depth is equal to user specified max
	if (m_max_tree_depth>0)
	{
		if (level==m_max_tree_depth)
			return node;
	}

	// case 4 : node size is less than user-specified min node size
	if (m_min_node_size>1)
	{
		if (num_vecs<m_min_node_size)
			return node;
	}

	// choose best attribute for splitting
	float64_t min_pv=CMath::MAX_REAL_NUMBER;
	SGVector<int32_t> cat_min;
	int32_t attr_min=-1;
	for (int32_t i=0;i<num_feats;i++)
	{
		SGVector<float64_t> feats(num_vecs);
		for (int32_t j=0;j<num_vecs;j++)
			feats[j]=mat(i,j);

		float64_t pv=0;
		SGVector<int32_t> cat;
		if (m_feature_types[i]==0)
			cat=merge_categories_nominal(feats,labels_vec,weights,pv);
		else if (m_feature_types[i]==1)
			cat=merge_categories_ordinal(feats,labels_vec,weights,pv);
		else
			SG_ERROR("feature type supported are 0(nominal) and 1(ordinal). m_feature_types[%d] is set %d\n",i,m_feature_types[i])

		if (pv<min_pv)
		{
			min_pv=pv;
			attr_min=i;
			cat_min=cat;
		}
	}

	if (min_pv>m_alpha_split)
		return node;

	// split
	SGVector<float64_t> ufeats_best(num_vecs);
	for (int32_t i=0;i<num_vecs;i++)
		ufeats_best[i]=mat(attr_min,i);

	int32_t unum=ufeats_best.unique(ufeats_best.vector,ufeats_best.vlen);
	for (int32_t i=0;i<cat_min.vlen;i++)
	{
		if (cat_min[i]!=i)
			continue;

		CDynamicArray<int32_t>* feat_index=new CDynamicArray<int32_t>();
		for (int32_t j=0;j<num_vecs;j++)
		{
			for (int32_t k=0;k<unum;k++)
			{
				if (mat(attr_min,j)==ufeats_best[k])
				{
					if (cat_min[k]==i)
						feat_index->push_back(j);
				}
			}
		}

		SGVector<int32_t> subset(feat_index->get_num_elements());
		SGVector<float64_t> subweights(feat_index->get_num_elements());
		for (int32_t j=0;j<feat_index->get_num_elements();j++)
		{
			subset[j]=feat_index->get_element(j);
			subweights[j]=weights[feat_index->get_element(j)];
		}

		data->add_subset(subset);
		labels->add_subset(subset);
		node_t* child=CHAIDtrain(data,subweights,labels,level+1);
		node->add_child(child);

		node->data.attribute_id=attr_min;
		int32_t c=0;
		SGVector<int32_t> feat_class=cat_min.clone();
		for (int32_t j=0;j<feat_class.vlen;j++)
		{
			if (feat_class[j]!=j)
			{
				continue;
			}
			else if (j==c)
			{
				c++;
				continue;
			}

			for (int32_t k=j;k<feat_class.vlen;k++)
			{
				if (feat_class[k]==j)
					feat_class[k]=c;
			}

			c++;
		}

		node->data.feature_class=feat_class;
		node->data.distinct_features=SGVector<float64_t>(unum);
		for (int32_t j=0;j<unum;j++)
			node->data.distinct_features[j]=ufeats_best[j];

		SG_UNREF(feat_index);
		data->remove_subset();
		labels->remove_subset();
	}

	return node;
}

SGVector<int32_t> CCHAIDTree::merge_categories_ordinal(SGVector<float64_t> feats, SGVector<float64_t> labels, 
							SGVector<float64_t> weights, float64_t &pv)
{
	SGVector<float64_t> ufeats=feats.clone();
	int32_t inum_cat=ufeats.unique(ufeats.vector,ufeats.vlen);
	SGVector<int32_t> cat(inum_cat);
	cat.range_fill(0);

	if (inum_cat==1)
	{
		pv=1.0;
		return cat;
	}

	bool missing=false;
	if (ufeats[inum_cat-1]==MISSING)
	{
		missing=true;
		inum_cat--;
	}

	int32_t fnum_cat=inum_cat; 

	// if chosen attribute (MISSING excluded) has 1 category only
	if (inum_cat==1)
	{
		pv=adjusted_p_value(p_value(feats,labels,weights),2,2,1,true);
		return cat;
	}

	while(true)
	{
		if (fnum_cat==2)
			break;

		// scan all allowable pairs of categories to find most similar one 
		int32_t cat_index_max=-1;
		float64_t max_merge_pv=CMath::MIN_REAL_NUMBER;
		for (int32_t i=0;i<inum_cat-1;i++)
		{
			if (cat[i]==cat[i+1])
				continue;

			int32_t cat_index=i;

			// compute p-value
			CDynamicArray<int32_t>* feat_index=new CDynamicArray<int32_t>();
			CDynamicArray<int32_t>* feat_cat=new CDynamicArray<int32_t>();
			for (int32_t j=0;j<feats.vlen;j++)
			{
				for (int32_t k=0;k<inum_cat;k++)
				{
					if (feats[j]==ufeats[k])
					{
						if (cat[k]==cat[cat_index])
						{
							feat_index->push_back(j);
							feat_cat->push_back(cat[cat_index]);
						}
						else if (cat[k]==cat[cat_index+1])
						{
							feat_index->push_back(j);
							feat_cat->push_back(cat[cat_index+1]);
						}
					}
				}
			}

			SGVector<float64_t> subfeats(feat_index->get_num_elements());
			SGVector<float64_t> sublabels(feat_index->get_num_elements());
			SGVector<float64_t> subweights(feat_index->get_num_elements());
			for (int32_t j=0;j<feat_index->get_num_elements();j++)
			{
				subfeats[j]=feat_cat->get_element(j);
				sublabels[j]=labels[feat_index->get_element(j)];
				subweights[j]=weights[feat_index->get_element(j)];
			}

			float64_t pvalue=p_value(subfeats,sublabels,subweights);
			if (pvalue>max_merge_pv)
			{
				max_merge_pv=pvalue;
				cat_index_max=cat_index;
			}

			SG_UNREF(feat_index);
			SG_UNREF(feat_cat);
		}

		if (max_merge_pv>m_alpha_merge)
		{
			// merge
			int32_t cat2=cat[cat_index_max+1];
			for (int32_t i=cat_index_max+1;i<inum_cat;i++)
			{
				if (cat2==cat[i])
					cat[i]=cat[cat_index_max];
				else
					break;
			}

			fnum_cat--;
		}
		else
		{
			break;
		}
	}

	SGVector<float64_t> feats_cat(feats.vlen);
	for (int32_t i=0;i<feats.vlen;i++)
	{
		if (feats[i]==MISSING)
		{
			feats_cat[i]=MISSING;
			continue;
		}

		for (int32_t j=0;j<inum_cat;j++)
		{
			if (feats[i]==ufeats[j])
				feats_cat[i]=cat[j];
		}
	}

	if (missing)
	{
		bool merged=handle_missing_ordinal(cat,feats_cat,labels,weights);
		if (!merged)
			fnum_cat+=1;

		pv=adjusted_p_value(p_value(feats_cat,labels,weights),inum_cat+1,fnum_cat,1,true);
	}
	else
	{
		pv=adjusted_p_value(p_value(feats_cat,labels,weights),inum_cat,fnum_cat,1,false);
	}

	return cat;
}

SGVector<int32_t> CCHAIDTree::merge_categories_nominal(SGVector<float64_t> feats, SGVector<float64_t> labels, 
								SGVector<float64_t> weights, float64_t &pv)
{
	SGVector<float64_t> ufeats=feats.clone();
	int32_t inum_cat=ufeats.unique(ufeats.vector,ufeats.vlen);
	int32_t fnum_cat=inum_cat; 

	SGVector<int32_t> cat(inum_cat);
	cat.range_fill(0);

	// if chosen attribute X(feats here) has 1 category only
	if (inum_cat==1)
	{
		pv=1.0;
		return cat;
	}

	while(true)
	{
		if (fnum_cat==2)
			break;

		// assimilate all category labels left
		CDynamicArray<int32_t>* leftcat=new CDynamicArray<int32_t>();
		for (int32_t i=0;i<cat.vlen;i++)
		{
			if (cat[i]==i)
				leftcat->push_back(i);
		}

		// consider all pairs for merging
		float64_t max_merge_pv=CMath::MIN_REAL_NUMBER;
		int32_t cat1_max=-1;
		int32_t cat2_max=-1;
		for (int32_t i=0;i<leftcat->get_num_elements()-1;i++)
		{
			for (int32_t j=i+1;j<leftcat->get_num_elements();j++)
			{
				CDynamicArray<int32_t>* feat_index=new CDynamicArray<int32_t>();
				CDynamicArray<int32_t>* feat_cat=new CDynamicArray<int32_t>();
				for (int32_t k=0;k<feats.vlen;k++)
				{
					for (int32_t l=0;l<inum_cat;l++)
					{
						if (feats[k]==ufeats[l])
						{
							if (cat[l]==leftcat->get_element(i))
							{
								feat_index->push_back(k);
								feat_cat->push_back(leftcat->get_element(i));
							}
							else if (cat[l]==leftcat->get_element(j))
							{
								feat_index->push_back(k);
								feat_cat->push_back(leftcat->get_element(j));
							}
						}
					}
				}

				SGVector<float64_t> subfeats(feat_index->get_num_elements());
				SGVector<float64_t> sublabels(feat_index->get_num_elements());
				SGVector<float64_t> subweights(feat_index->get_num_elements());
				for (int32_t k=0;k<feat_index->get_num_elements();k++)
				{
					subfeats[k]=feat_cat->get_element(k);
					sublabels[k]=labels[feat_index->get_element(k)];
					subweights[k]=weights[feat_index->get_element(k)];
				}

				float64_t pvalue=p_value(subfeats,sublabels,subweights);
				if (pvalue>max_merge_pv)
				{
					max_merge_pv=pvalue;
					cat1_max=leftcat->get_element(i);
					cat2_max=leftcat->get_element(j);
				}

				SG_UNREF(feat_index);
				SG_UNREF(feat_cat);
			}
		}

		SG_UNREF(leftcat);

		if (max_merge_pv>m_alpha_merge)
		{
			// merge
			for (int32_t i=0;i<cat.vlen;i++)
			{
				if (cat2_max==cat[i])
					cat[i]=cat1_max;
			}

			fnum_cat--;
		}
		else
		{
			break;
		}
	}

	SGVector<float64_t> feats_cat(feats.vlen);
	for (int32_t i=0;i<feats.vlen;i++)
	{
		for (int32_t j=0;j<inum_cat;j++)
		{
			if (feats[i]==ufeats[j])
				feats_cat[i]=cat[j];
		}
	}

	pv=adjusted_p_value(p_value(feats_cat,labels,weights),inum_cat,fnum_cat,0,false);
	return cat;
}

bool CCHAIDTree::handle_missing_ordinal(SGVector<int32_t> cat, SGVector<float64_t> feats, SGVector<float64_t> labels,
									 		SGVector<float64_t> weights)
{
	// assimilate category indices other than missing (last cell of cat vector stores category index for missing)
	// sanity check
	REQUIRE(cat[cat.vlen-1]==cat.vlen-1,"last category is expected to be stored for MISSING. Hence it is expected to be un-merged\n")
	CDynamicArray<int32_t>* cat_ind=new CDynamicArray<int32_t>();
	for (int32_t i=0;i<cat.vlen-1;i++)
	{
		if (cat[i]==i)
			cat_ind->push_back(i);
	}

	// find most similar category to MISSING
	float64_t max_pv_pair=CMath::MIN_REAL_NUMBER;
	int32_t cindex_max=-1;
	for (int32_t i=0;i<cat_ind->get_num_elements();i++)
	{
		CDynamicArray<int32_t>* feat_index=new CDynamicArray<int32_t>();
		for (int32_t j=0;j<feats.vlen;j++)
		{
			if ((feats[j]==cat_ind->get_element(i)) || feats[j]==MISSING)
				feat_index->push_back(j);
		}

		SGVector<float64_t> subfeats(feat_index->get_num_elements());
		SGVector<float64_t> sublabels(feat_index->get_num_elements());
		SGVector<float64_t> subweights(feat_index->get_num_elements());
		for (int32_t j=0;j<feat_index->get_num_elements();j++)
		{
			subfeats[j]=feats[feat_index->get_element(j)];
			sublabels[j]=labels[feat_index->get_element(j)];
			subweights[j]=weights[feat_index->get_element(j)];
		}

		float64_t pvalue=p_value(subfeats,sublabels,subweights);
		if (pvalue>max_pv_pair)
		{
			max_pv_pair=pvalue;
			cindex_max=cat_ind->get_element(i);
		}

		SG_UNREF(feat_index);
	}

	// compare if MISSING being merged is better than not being merged 
	SGVector<float64_t> feats_copy(feats.vlen);
	for (int32_t i=0;i<feats.vlen;i++)
	{
		if (feats[i]==MISSING)
			feats_copy[i]=cindex_max;
		else
			feats_copy[i]=feats[i];
	}

	float64_t pv_merged=p_value(feats_copy, labels, weights);
	float64_t pv_unmerged=p_value(feats, labels, weights);
	if (pv_merged>pv_unmerged)
	{
		cat[cat.vlen-1]=cindex_max;
		for (int32_t i=0;i<feats.vlen;i++)
		{
			if (feats[i]==MISSING)
				feats[i]=cindex_max;
		}

		return true;
	}

	return false;
}

float64_t CCHAIDTree::adjusted_p_value(float64_t up_value, int32_t inum_cat, int32_t fnum_cat, int32_t ft, bool is_missing)
{

	if (inum_cat==fnum_cat)
		return up_value;

	switch (ft)
	{
		case 0:
		{
			float64_t sum=0.;
			for (int32_t v=0;v<fnum_cat;v++)
			{
				float64_t lterm=inum_cat*CMath::log(fnum_cat-v);
				for (int32_t j=1;j<=v;j++)
					lterm-=CMath::log(j);

				for (int32_t j=1;j<=fnum_cat-v;j++)
					lterm-=CMath::log(j);

				if (v%2==0)
					sum+=CMath::exp(lterm);
				else
					sum-=CMath::exp(lterm);
			}

			return sum*up_value;
		}
		case 1:
		{
			if (!is_missing)
				return CMath::nchoosek(inum_cat-1,fnum_cat-1)*up_value;
			else
				return up_value*(CMath::nchoosek(inum_cat-2,fnum_cat-2)+fnum_cat*CMath::nchoosek(inum_cat-2,fnum_cat-1));
		}
		default:
			SG_ERROR("Feature type must be either 0 (nominal) or 1 (ordinal). It is currently set as %d\n",ft)
	}

	return 0.0;
}

float64_t CCHAIDTree::p_value(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights)
{
	switch (m_dependent_vartype)
	{
		case 0:
		{
			int32_t r=0;
			int32_t c=0;
			float64_t x2=pchi2_statistic(feat,labels,weights,r,c);
			return 1-CStatistics::chi2_cdf(x2,(r-1)*(c-1));
		}
		case 1:
		{
			int32_t r=0;
			int32_t c=0;
			float64_t h2=likelihood_ratio_statistic(feat,labels,weights,r,c);
			return 1-CStatistics::chi2_cdf(h2,(r-1));
		}
		case 2:
		{
			int32_t nf=feat.vlen;
			int32_t num_cat=0;
			float64_t f=anova_f_statistic(feat,labels,weights,num_cat);
			return 1-CStatistics::fdistribution_cdf(f,num_cat-1,nf-num_cat);
		}
		default:
			SG_ERROR("Dependent variable type must be either 0 or 1 or 2. It is currently set as %d\n",m_dependent_vartype)
	}

	return -1.0;
}

float64_t CCHAIDTree::anova_f_statistic(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights, int32_t &r)
{
	// compute y_bar
	float64_t y_bar=0.;
	for (int32_t i=0;i<labels.vlen;i++)
		y_bar+=labels[i]*weights[i];

	y_bar/=weights.sum(weights);

	SGVector<float64_t> ufeat=feat.clone();
	r=ufeat.unique(ufeat.vector,ufeat.vlen);

	// compute y_i_bar
	SGVector<float64_t> numer(r);
	SGVector<float64_t> denom(r);
	numer.zero();
	denom.zero();
	for (int32_t n=0;n<feat.vlen;n++)
	{
		for (int32_t i=0;i<r;i++)
		{
			if (feat[n]==ufeat[i])
			{
				numer[i]+=weights[n]*labels[n];
				denom[i]+=weights[n];
				break;
			}
		}
	}

	// compute f statistic
	float64_t nu=0.;
	float64_t de=0.;
	for (int32_t i=0;i<r;i++)
	{
		for (int32_t n=0;n<feat.vlen;n++)
		{
			if (feat[n]==ufeat[i])
			{
				nu+=weights[n]*CMath::pow(((numer[i]/denom[i])-y_bar),2);
				de+=weights[n]*CMath::pow((labels[n]-(numer[i]/denom[i])),2);
			}
		}
	}

	nu/=(r-1.0);
	de/=(feat.vlen-r-0.f);

	return nu/de;
}

float64_t CCHAIDTree::likelihood_ratio_statistic(SGVector<float64_t> feat, SGVector<float64_t> labels, 
						SGVector<float64_t> weights, int32_t &r, int32_t &c)
{
	SGVector<float64_t> ufeat=feat.clone();
	SGVector<float64_t> ulabels=labels.clone();
	r=ufeat.unique(ufeat.vector,ufeat.vlen);
	c=ulabels.unique(ulabels.vector,ulabels.vlen);

	// contingency table, weight table
	SGMatrix<int32_t> ct(r,c);
	ct.zero();
	SGMatrix<float64_t> wt(r,c);
	wt.zero();
	for (int32_t i=0;i<feat.vlen;i++)
	{
		// calculate row
		int32_t row=-1;
		for (int32_t j=0;j<r;j++)
		{
			if (feat[i]==ufeat[j])
			{
				row=j;
				break;
			}
		}

		// calculate col
		int32_t col=-1;
		for (int32_t j=0;j<c;j++)
		{
			if (labels[i]==ulabels[j])
			{
				col=j;
				break;
			}
		}

		ct(row,col)++;
		wt(row,col)+=weights[i];
	}

	SGMatrix<float64_t> expmat_indep=expected_cf_indep_model(ct,wt);

	SGVector<float64_t> score(c);
	score.range_fill(1.0);
	SGMatrix<float64_t> expmat_row_effects=expected_cf_row_effects_model(ct,wt,score);

	float64_t ret=0.;
	for (int32_t i=0;i<r;i++)
	{
		for (int32_t j=0;j<c;j++)
			ret+=expmat_row_effects(i,j)*CMath::log(expmat_row_effects(i,j)/expmat_indep(i,j));
	}

	return 2*ret;
}

float64_t CCHAIDTree::pchi2_statistic(SGVector<float64_t> feat, SGVector<float64_t> labels, SGVector<float64_t> weights,
												int32_t &r, int32_t &c)
{
	SGVector<float64_t> ufeat=feat.clone();
	SGVector<float64_t> ulabels=labels.clone();
	r=ufeat.unique(ufeat.vector,ufeat.vlen);
	c=ulabels.unique(ulabels.vector,ulabels.vlen);

	// contingency table, weight table
	SGMatrix<int32_t> ct(r,c);
	ct.zero();
	SGMatrix<float64_t> wt(r,c);
	wt.zero();
	for (int32_t i=0;i<feat.vlen;i++)
	{
		// calculate row
		int32_t row=-1;
		for (int32_t j=0;j<r;j++)
		{
			if (feat[i]==ufeat[j])
			{
				row=j;
				break;
			}
		}

		// calculate col
		int32_t col=-1;
		for (int32_t j=0;j<c;j++)
		{
			if (labels[i]==ulabels[j])
			{
				col=j;
				break;
			}
		}

		ct(row,col)++;
		wt(row,col)+=weights[i];
	}

	SGMatrix<float64_t> expected_cf=expected_cf_indep_model(ct,wt);

	float64_t ret=0.;
	for (int32_t i=0;i<r;i++)
	{
		for (int32_t j=0;j<c;j++)
			ret+=CMath::pow((ct(i,j)-expected_cf(i,j)),2)/expected_cf(i,j);
	}

	return ret;
}

SGMatrix<float64_t> CCHAIDTree::expected_cf_row_effects_model(SGMatrix<int32_t> ct, SGMatrix<float64_t> wt, SGVector<float64_t> score)
{
	int32_t r=ct.num_rows;
	int32_t c=ct.num_cols;

	// compute row sum(n_i.'s) and column sum(n_.j's)
	SGVector<int32_t> row_sum(r);
	SGVector<int32_t> col_sum(c);
	for (int32_t i=0;i<r;i++)
	{
		int32_t sum=0;
		for (int32_t j=0;j<c;j++)
			sum+=ct(i,j);

		row_sum[i]=sum;
	}
	for (int32_t i=0;i<c;i++)
	{
		int32_t sum=0;
		for (int32_t j=0;j<r;j++)
			sum+=ct(j,i);

		col_sum[i]=sum;
	}

	// compute s_bar
	float64_t numer=0.;
	float64_t denom=0.;
	for (int32_t j=0;j<c;j++)
	{
		float64_t w_j=0.;
		for (int32_t i=0;i<r;i++)
			w_j+=wt(i,j);

		denom+=w_j;
		numer+=w_j*score[j];
	}

	float64_t s_bar=numer/denom;

	// element-wise normalize and invert weight matrix w_ij(new)=n_ij/w_ij(old)
	for (int32_t i=0;i<r;i++)
	{
		for (int32_t j=0;j<c;j++)
			wt(i,j)=(ct(i,j)-0.f)/wt(i,j);
	}

	SGMatrix<float64_t> m_k=wt.clone();
	SGVector<float64_t> alpha(r);
	SGVector<float64_t> beta(c);
	SGVector<float64_t> gamma(r);
	alpha.fill_vector(alpha.vector,alpha.vlen,1.0);
	beta.fill_vector(beta.vector,beta.vlen,1.0);
	gamma.fill_vector(gamma.vector,gamma.vlen,1.0);
	float64_t epsilon=1e-8;
	while(true)
	{
		// update alpha
		for (int32_t i=0;i<r;i++)
		{
			float64_t sum=0.;
			for (int32_t j=0;j<c;j++)
				sum+=m_k(i,j);

			alpha[i]*=(row_sum[i]-0.f)/sum;
		}

		// update beta
		for (int32_t j=0;j<c;j++)
		{
			float64_t sum=0.;
			for (int32_t i=0;i<r;i++)
				sum+=wt(i,j)*alpha[i]*CMath::pow(gamma[i],(score[j]-s_bar));

			beta[j]=(col_sum[j]-0.f)/sum;
		}

		// compute g_i for updating gamma
		SGVector<float64_t> g(r);
		SGMatrix<float64_t> m_star(r,c);
		for (int32_t i=0;i<r;i++)
		{
			for (int32_t j=0;j<c;j++)
				m_star(i,j)=wt(i,j)*alpha[i]*beta[j]*CMath::pow(gamma[i],score[j]-s_bar);
		}

		for (int32_t i=0;i<r;i++)
		{
			numer=0.;
			denom=0.;
			for (int32_t j=0;j<c;j++)
			{
				numer+=(score[j]-s_bar)*(ct(i,j)-m_star(i,j));
				denom+=CMath::pow((score[j]-s_bar),2)*m_star(i,j);
			}

			g[i]=1+numer/denom;
		}

		// update gamma
		for (int32_t i=0;i<r;i++)
			gamma[i]=(g[i]>0)?gamma[i]*g[i]:gamma[i];

		// update m_k
		SGMatrix<float64_t> m_kplus(r,c);
		float64_t max_diff=0.;
		for (int32_t i=0;i<r;i++)
		{
			for (int32_t j=0;j<c;j++)
			{
				m_kplus(i,j)=wt(i,j)*alpha[i]*beta[j]*CMath::pow(gamma[i],(score[j]-s_bar));
				float64_t abs_diff=CMath::abs(m_kplus(i,j)-m_k(i,j));
				if (abs_diff>max_diff)
					max_diff=abs_diff;
			}
		}

		m_k=m_kplus;
		if (max_diff<epsilon)
			break;
	}

	return m_k;
}

SGMatrix<float64_t> CCHAIDTree::expected_cf_indep_model(SGMatrix<int32_t> ct, SGMatrix<float64_t> wt)
{
	int32_t r=ct.num_rows;
	int32_t c=ct.num_cols;

	// compute row sum(n_i.'s) and column sum(n_.j's)
	SGVector<int32_t> row_sum(r);
	SGVector<int32_t> col_sum(c);
	for (int32_t i=0;i<r;i++)
	{
		int32_t sum=0;
		for (int32_t j=0;j<c;j++)
			sum+=ct(i,j);

		row_sum[i]=sum;
	}
	for (int32_t i=0;i<c;i++)
	{
		int32_t sum=0;
		for (int32_t j=0;j<r;j++)
			sum+=ct(j,i);

		col_sum[i]=sum;
	}

	SGMatrix<float64_t> ret(r,c);

	// if all weights are 1 - m_ij=n_i.*n_.j/n..
	if (!m_weights_set)
	{
		int32_t total_sum=(r<=c)?row_sum.sum(row_sum):col_sum.sum(col_sum);

		for (int32_t i=0;i<r;i++)
		{
			for (int32_t j=0;j<c;j++)
				ret(i,j)=(row_sum[i]*col_sum[j]-0.f)/(total_sum-0.f);
		}
	}
	else
	{
		// element-wise normalize and invert weight matrix w_ij(new)=n_ij/w_ij(old)
		for (int32_t i=0;i<r;i++)
		{
			for (int32_t j=0;j<c;j++)
				wt(i,j)=(ct(i,j)-0.f)/wt(i,j);
		}

		// iteratively estimate mij
		SGMatrix<float64_t> m_k=wt.clone();
		SGVector<float64_t> alpha(r);
		SGVector<float64_t> beta(c);
		alpha.fill_vector(alpha.vector,alpha.vlen,1.0);
		beta.fill_vector(beta.vector,beta.vlen,1.0);
		float64_t epsilon=1e-8;
		while (true)
		{
			// update alpha
			for (int32_t i=0;i<r;i++)
			{
				float64_t sum=0.;
				for (int32_t j=0;j<c;j++)
					sum+=m_k(i,j);

				alpha[i]*=(row_sum[i]-0.f)/sum;
			}

			// update beta
			for (int32_t j=0;j<c;j++)
			{
				float64_t sum=0.;
				for (int32_t i=0;i<r;i++)
					sum+=wt(i,j)*alpha[i];

				beta[j]=(col_sum[j]-0.f)/sum;
			}

			// update m_k
			SGMatrix<float64_t> m_kplus(r,c);
			float64_t max_diff=0.0;
			for (int32_t i=0;i<r;i++)
			{
				for (int32_t j=0;j<c;j++)
				{
					m_kplus(i,j)=wt(i,j)*alpha[i]*beta[j];
					float64_t abs_diff=CMath::abs(m_kplus(i,j)-m_k(i,j));
					if (abs_diff>max_diff)
						max_diff=abs_diff;
				}
			}

			m_k=m_kplus;

			if (max_diff<epsilon)
				break;
		}

		ret=m_k;
	}

	return ret;
}

float64_t CCHAIDTree::sum_of_squared_deviation(SGVector<float64_t> lab, SGVector<float64_t> weights, float64_t &mean)
{
	mean=0;
	float64_t total_weight=0;
	for (int32_t i=0;i<lab.vlen;i++)
	{
		mean+=lab[i]*weights[i];
		total_weight+=weights[i];
	}

	mean/=total_weight;
	float64_t dev=0;
	for (int32_t i=0;i<lab.vlen;i++)
		dev+=weights[i]*(lab[i]-mean)*(lab[i]-mean);

	return dev;
}

bool CCHAIDTree::continuous_to_ordinal(SGMatrix<float64_t> featmat)
{
	// assimilate continuous breakpoints
	int32_t count_cont=0;
	for (int32_t i=0;i<featmat.num_rows;i++)
	{
		if (m_feature_types[i]==2)
			count_cont++;
	}

	if (count_cont==0)
		return false;

	SGVector<int32_t> cont_ind(count_cont);
	int32_t ci=0;
	for (int32_t i=0;i<featmat.num_rows;i++)
	{
		if (m_feature_types[i]==2)
			cont_ind[ci++]=i;
	}

	// form breakpoints matrix
	m_cont_breakpoints=SGMatrix<float64_t>(m_num_breakpoints,count_cont);
	int32_t bin_size=featmat.num_cols/m_num_breakpoints;
	for (int32_t i=0;i<count_cont;i++)
	{
		int32_t left=featmat.num_cols%m_num_breakpoints;
		int32_t end_pt=-1;

		SGVector<float64_t> values(featmat.num_cols);
		for (int32_t j=0;j<values.vlen;j++)
			values[j]=featmat(cont_ind[i],j);

		values.qsort();

		for (int32_t j=0;j<m_num_breakpoints;j++)
		{
			if (left>0)
			{
				left--;
				end_pt+=bin_size+1;
				m_cont_breakpoints(j,i)=values[end_pt];
			}
			else
			{
				end_pt+=bin_size;
				m_cont_breakpoints(j,i)=values[end_pt];
			}
		}
	}

	// update data matrix
	for (int32_t i=0;i<count_cont;i++)
	{
		for (int32_t j=0;j<featmat.num_cols;j++)
		{
			// find right bin
			for (int32_t k=0;k<m_num_breakpoints;k++)
			{
				if (featmat(cont_ind[i],j)<m_cont_breakpoints(i,k))
					featmat(cont_ind[i],j)=m_cont_breakpoints(i,k);
			} 
		}
	}

	return true;
}

void CCHAIDTree::init()
{
	m_feature_types=SGVector<int32_t>();
	m_weights=SGVector<float64_t>();
	m_dependent_vartype=0;
	m_weights_set=false;
	m_max_tree_depth=0;
	m_min_node_size=0;
	m_alpha_merge=0.05;
	m_alpha_split=0.05;
	m_cont_breakpoints=SGMatrix<float64_t>();
	m_num_breakpoints=5;

	SG_ADD(&m_weights,"m_weights", "weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_set,"m_weights_set", "weights set", MS_NOT_AVAILABLE);
	SG_ADD(&m_feature_types,"m_feature_types", "feature types", MS_NOT_AVAILABLE);
	SG_ADD(&m_dependent_vartype,"m_dependent_vartype", "dependent variable type", MS_NOT_AVAILABLE);
	SG_ADD(&m_max_tree_depth,"m_max_tree_depth", "max tree depth", MS_NOT_AVAILABLE);
	SG_ADD(&m_min_node_size,"m_min_node_size", "min node size", MS_NOT_AVAILABLE);
	SG_ADD(&m_alpha_merge,"m_alpha_merge", "alpha-merge", MS_NOT_AVAILABLE);
	SG_ADD(&m_alpha_split,"m_alpha_split", "alpha-split", MS_NOT_AVAILABLE);
	SG_ADD(&m_cont_breakpoints,"m_cont_breakpoints", "breakpoints in continuous attributes", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_breakpoints,"m_num_breakpoints", "number of breakpoints", MS_NOT_AVAILABLE);
}
