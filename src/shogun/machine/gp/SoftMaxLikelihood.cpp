/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar, Wu Lin
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
 *
 * Code adapted from
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
 * and
 * GPstuff - Gaussian process models for Bayesian analysis
 * http://becs.aalto.fi/en/research/bayes/gpstuff/
 *
 * The reference pseudo code is the algorithm 3.4 of the GPML textbook
 */

#include <shogun/machine/gp/SoftMaxLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/distributions/classical/GaussianDistribution.h>

using namespace shogun;
using namespace Eigen;

CSoftMaxLikelihood::CSoftMaxLikelihood() : CLikelihoodModel()
{
	init();
}

CSoftMaxLikelihood::~CSoftMaxLikelihood()
{
}

void CSoftMaxLikelihood::init()
{
	m_num_samples=10000;
	SG_ADD(&m_num_samples, "num_samples",
		"Number of samples to be generated",
		MS_NOT_AVAILABLE);
}

SGVector<float64_t> CSoftMaxLikelihood::get_log_probability_f(const CLabels* lab,
					 SGVector<float64_t> func) const
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_MULTICLASS,
			"Labels must be type of CMulticlassLabels\n")

	SGVector<int32_t> labels=((CMulticlassLabels*) lab)->get_int_labels();
	for (int32_t i=0;i<labels.vlen;i++)
		REQUIRE(((labels[i]>-1)&&(labels[i]<func.vlen/labels.vlen)),
		 "Labels must be between 0 and C(ie %d here). Currently labels[%d] is"
		"%d\n",func.vlen/labels.vlen,i,labels[i]);

	// labels.vlen=num_rows  func.vlen/num_rows=num_cols
	Map<MatrixXd> eigen_f(func.vector,labels.vlen,func.vlen/labels.vlen);

	// log_sum_exp trick
	VectorXd max_coeff=eigen_f.rowwise().maxCoeff();
	eigen_f=eigen_f.array().colwise()-max_coeff.array();
	VectorXd log_sum_exp=((eigen_f.array().exp()).rowwise().sum()).array().log();
	log_sum_exp=log_sum_exp+max_coeff;

	// restore original matrix
	eigen_f=eigen_f.array().colwise()+max_coeff.array();

	SGVector<float64_t> ret=SGVector<float64_t>(labels.vlen);
	Map<VectorXd> eigen_ret(ret.vector,ret.vlen);

	for (int32_t i=0;i<labels.vlen;i++)
		eigen_ret(i)=eigen_f(i,labels[i]);

	eigen_ret=eigen_ret-log_sum_exp;

	return ret;
}

SGVector<float64_t> CSoftMaxLikelihood::get_log_probability_derivative_f(
		const CLabels* lab, SGVector<float64_t> func, index_t i) const
{
	int32_t num_rows=lab->get_num_labels();
	int32_t num_cols=func.vlen/num_rows;
	SGMatrix<float64_t> f=SGMatrix<float64_t>(func.vector,num_rows,num_cols,false);

	if (i==1)
		return get_log_probability_derivative1_f(lab,f);
	else if (i==2)
		return get_log_probability_derivative2_f(f);
	else
		return get_log_probability_derivative3_f(f);
}

SGVector<float64_t> CSoftMaxLikelihood::get_log_probability_derivative1_f(
			const CLabels* lab, SGMatrix<float64_t> func) const
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_MULTICLASS,
			"Labels must be type of CMulticlassLabels\n")
	REQUIRE(lab->get_num_labels()==func.num_rows, "Number of labels must match "
			"number of vectors in function\n")

	SGVector<int32_t> labels=((CMulticlassLabels*) lab)->get_int_labels();
	for (int32_t i=0;i<labels.vlen;i++)
		REQUIRE(((labels[i]>-1)&&(labels[i]<func.num_cols)),
		 "Labels must be between 0 and C(ie %d here). Currently labels[%d] is"
		"%d\n",func.num_cols,i,labels[i]);

	SGVector<float64_t> ret=SGVector<float64_t>(func.num_rows*func.num_cols);
	memcpy(ret.vector,func.matrix,func.num_rows*func.num_cols*sizeof(float64_t));

	//pi
	Map<MatrixXd> eigen_ret(ret.vector,func.num_rows,func.num_cols);

	// with log_sum_exp trick
	VectorXd max_coeff=eigen_ret.rowwise().maxCoeff();
	eigen_ret=eigen_ret.array().colwise()-max_coeff.array();
	VectorXd log_sum_exp=((eigen_ret.array().exp()).rowwise().sum()).array().log();
	eigen_ret=(eigen_ret.array().colwise()-log_sum_exp.array()).exp();

	// without log_sum_exp trick
	//eigen_ret=eigen_ret.array().exp();
	//VectorXd tmp=eigen_ret.rowwise().sum();
	//eigen_ret=eigen_ret.array().colwise()/tmp.array();

	MatrixXd y=MatrixXd::Zero(func.num_rows,func.num_cols);

	for (int32_t i=0;i<labels.vlen;i++)
		y(i,labels[i])=1.;

	eigen_ret=y-eigen_ret;

	return ret;
}

SGVector<float64_t> CSoftMaxLikelihood::get_log_probability_derivative2_f(SGMatrix<float64_t> func) const
{
	SGVector<float64_t> ret=SGVector<float64_t>(func.num_rows*func.num_cols*func.num_cols);
	Map<MatrixXd> eigen_ret(ret.vector,func.num_rows*func.num_cols,func.num_cols);

	Map<MatrixXd> eigen_f(func.matrix,func.num_rows,func.num_cols);

	MatrixXd f1= eigen_f.array().exp();
	VectorXd tmp=f1.rowwise().sum();
	f1=f1.array().colwise()/tmp.array();

	for (int32_t i=0;i<eigen_f.rows();i++)
	{
		eigen_ret.block(i*eigen_f.cols(),0,eigen_f.cols(),eigen_f.cols())=
						f1.transpose().col(i)*f1.row(i);
		VectorXd D=eigen_ret.block(i*eigen_f.cols(),0,eigen_f.cols(),eigen_f.cols())
								.diagonal().array().sqrt();
		eigen_ret.block(i*eigen_f.cols(),0,eigen_f.cols(),eigen_f.cols())-=
								MatrixXd(D.asDiagonal());
	}

	return ret;
}

SGVector<float64_t> CSoftMaxLikelihood::get_log_probability_derivative3_f(SGMatrix<float64_t>
										 func) const
{
	SGVector<float64_t> ret=SGVector<float64_t>(CMath::pow(func.num_cols,3)*func.num_rows);

	Map<MatrixXd> eigen_f(func.matrix,func.num_rows,func.num_cols);

	MatrixXd f1= eigen_f.array().exp();
	VectorXd tmp=f1.rowwise().sum();
	f1=f1.array().colwise()/tmp.array();

	for (int32_t i=0;i<func.num_rows;i++)
	{
		for (int32_t c1=0;c1<func.num_cols;c1++)
		{
			for (int32_t c2=0;c2<func.num_cols;c2++)
			{
				for (int32_t c3=0;c3<func.num_cols;c3++)
				{
					float64_t sum_temp=0;
					if ((c1==c2) && (c2==c3))
						sum_temp+=f1(i,c1);
					if (c1==c2)
						sum_temp=sum_temp-f1(i,c1)*f1(i,c3);
					if (c1==c3)
						sum_temp=sum_temp-f1(i,c1)*f1(i,c2);
					if (c2==c3)
						sum_temp=sum_temp-f1(i,c1)*f1(i,c2);
					sum_temp+=2.0*f1(i,c1)*f1(i,c2)*f1(i,c3);

					ret[i*CMath::pow(func.num_cols,3)+
					c1*CMath::pow(func.num_cols,2)+c2*func.num_cols+c3]=sum_temp;
				}
			}
		}
	}

	return ret;
}

void CSoftMaxLikelihood::set_num_samples(index_t num_samples)
{
	REQUIRE(num_samples>0, "Numer of samples (%d) should be positive\n",
		num_samples);
	m_num_samples=num_samples;
}

SGVector<float64_t> CSoftMaxLikelihood::predictive_helper(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels *lab, EMCSamplerType option) const
{
	const index_t C=s2.vlen/mu.vlen;
	const index_t n=mu.vlen/C;

	REQUIRE(n*C==mu.vlen, "Number of labels (%d) times number of classes (%d) must match "
		"number of elements(%d) in mu\n", n, C, mu.vlen);

	REQUIRE(n*C*C==s2.vlen, "Number of labels (%d) times second power of number of classes (%d*%d) must match "
		"number of elements(%d) in s2\n",n, C, C, s2.vlen);

	SGVector<index_t> y;

	if (lab)
	{
		REQUIRE(lab->get_label_type()==LT_MULTICLASS,
			"Labels must be type of CMulticlassLabels\n");

		const index_t n1=lab->get_num_labels();
		REQUIRE(n==n1, "Number of samples (%d) learned from mu and s2 must match "
			"number of labels(%d) in lab\n",n,n1);

		y=((CMulticlassLabels*) lab)->get_int_labels();
		for (index_t i=0;i<y.vlen;i++)
			REQUIRE(y[i]<C,"Labels must be between 0 and C(ie %d here). Currently lab[%d] is"
				"%d\n",C,i,y[i]);
	}
	else
	{
		y=SGVector<index_t>(n);
		y.set_const(C);
	}

	SGVector<float64_t> ret(mu.vlen);

	for(index_t idx=0; idx<n; idx++)
	{
		SGMatrix<float64_t> Sigma(s2.vector+idx*C*C, C, C, false);
		SGVector<float64_t> mean(mu.vector+idx*C, C, false);
		SGVector<float64_t> label(C);
		if (y[idx]<C)
		{
			label.set_const(0);
			label[y[idx]]=1.0;
		}
		else
		{
			label.set_const(1.0);
		}

		Map<VectorXd> eigen_ret_sub(ret.vector+idx*C, C);
		SGVector<float64_t> tmp=mc_sampler(m_num_samples,mean,Sigma,label);
		Map<VectorXd> eigen_tmp(tmp.vector, tmp.vlen);
		eigen_ret_sub=eigen_tmp;

		if (option==1)
		{
			Map<VectorXd> eigen_label(label.vector, label.vlen);
			eigen_ret_sub=eigen_ret_sub.array()*eigen_label.array()+(1-eigen_ret_sub.array())*(1-eigen_label.array());
		}
	}

	if (option==2)
	{
		Map<VectorXd> eigen_ret(ret.vector, ret.vlen);
		eigen_ret=eigen_ret.array()*(1-eigen_ret.array());
	}

	return ret;
}

SGVector<float64_t> CSoftMaxLikelihood::get_predictive_log_probabilities(
	SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels *lab)
{
	return predictive_helper(mu, s2, lab, MC_Probability);
}

SGVector<float64_t> CSoftMaxLikelihood::mc_sampler(index_t num_samples, SGVector<float64_t> mean,
	SGMatrix<float64_t> Sigma, SGVector<float64_t> y) const
{
	CGaussianDistribution *gen=new CGaussianDistribution (mean, Sigma);

	//category by samples
	SGMatrix<float64_t> samples=gen->sample(num_samples);
	Map<MatrixXd> eigen_samples(samples.matrix, samples.num_rows, samples.num_cols);

	MatrixXd my_samples=eigen_samples.array().exp();
	VectorXd sum_samples=my_samples.array().colwise().sum().transpose();
	MatrixXd normal_samples=(my_samples.array().rowwise()/sum_samples.array().transpose());
	VectorXd mean_samples=normal_samples.rowwise().mean();

	SGVector<float64_t>est(mean.vlen);
	Map<VectorXd> eigen_est(est.vector, est.vlen);

	//0 and 1 encoding
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	eigen_est=(mean_samples.array()*eigen_y.array())+(1-mean_samples.array())*(1-eigen_y.array());

	SG_UNREF(gen);

	return est;
}

SGVector<float64_t> CSoftMaxLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{

	return predictive_helper(mu, s2, lab, MC_Mean);
}

SGVector<float64_t> CSoftMaxLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	return predictive_helper(mu, s2, lab, MC_Variance);
}

#endif /* HAVE_EIGEN3 */
