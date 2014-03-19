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

#include <shogun/machine/gp/SoftMaxLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CSoftMaxLikelihood::CSoftMaxLikelihood() : CLikelihoodModel()
{
}

CSoftMaxLikelihood::~CSoftMaxLikelihood()
{
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
	eigen_f=eigen_f-max_coeff*MatrixXd::Ones(1,eigen_f.cols());
	VectorXd log_sum_exp=((eigen_f.array().exp()).rowwise().sum()).array().log();
	log_sum_exp=log_sum_exp+max_coeff;

	// restore original matrix
	eigen_f=eigen_f+max_coeff*MatrixXd::Ones(1,eigen_f.cols());

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

	Map<MatrixXd> eigen_ret(ret.vector,func.num_rows,func.num_cols);
	eigen_ret=eigen_ret.array().exp();
	eigen_ret=eigen_ret.cwiseQuotient(eigen_ret.rowwise().sum()*MatrixXd::Ones(1,func.num_cols));

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
	f1=f1.cwiseQuotient(f1.rowwise().sum()*MatrixXd::Ones(1,f1.cols()));

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
	f1=f1.cwiseQuotient(f1.rowwise().sum()*MatrixXd::Ones(1,f1.cols()));

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

#endif /* HAVE_EIGEN3 */
