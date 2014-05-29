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

const float64_t CCHAIDTree::MISSING=CMath::NOT_A_NUMBER;

CCHAIDTree::CCHAIDTree()
: CTreeMachine<CHAIDTreeNodeData>()
{
	init();
}

CCHAIDTree::~CCHAIDTree()
{
}

void CCHAIDTree::set_machine_problem_type(EProblemType mode)
{
	m_mode=mode;
}

bool CCHAIDTree::is_label_valid(CLabels* lab) const
{
	if (m_mode==PT_MULTICLASS && lab->get_label_type()==LT_MULTICLASS)
		return true;
	else if (m_mode==PT_REGRESSION && lab->get_label_type()==LT_REGRESSION)
		return true;
	else
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
	return true;
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
			SG_ERROR("Feature type must be either 0 (nominal) or 1 (ordinal). Its currently set as %d\n",ft)
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
			SG_ERROR("Dependent variable type must be either 0 or 1 or 2. Its currently set as %d\n",m_dependent_vartype)
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

void CCHAIDTree::init()
{
	m_feature_types=SGVector<int32_t>();
	m_weights=SGVector<float64_t>();
	m_dependent_vartype=-1;
	m_weights_set=false;

	SG_ADD(&m_weights,"m_weights", "weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_weights_set,"m_weights_set", "weights set", MS_NOT_AVAILABLE);
	SG_ADD(&m_feature_types,"m_feature_types", "feature types", MS_NOT_AVAILABLE);
	SG_ADD(&m_dependent_vartype,"m_dependent_vartype", "dependent variable type", MS_NOT_AVAILABLE);
}
