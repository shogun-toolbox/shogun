/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2014 Abhijeet Kislay
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/classifier/LDA.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;
using namespace shogun;

CLDA::CLDA(float64_t gamma, ELDAMethod method)
	:CLinearMachine()
{
	init();
	m_method=method;
	m_gamma=gamma;
}

CLDA::CLDA(float64_t gamma, CDenseFeatures<float64_t> *traindat,
			CLabels *trainlab, ELDAMethod method)
	:CLinearMachine(), m_gamma(gamma)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
	m_method=method;
	m_gamma=gamma;
}

void CLDA::init()
{
	m_method=AUTO_LDA;
	m_gamma=0;
	SG_ADD((machine_int_t*) &m_method, "m_method",
		"Method used for LDA calculation", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_gamma, "m_gamma",
		"Regularization parameter", MS_NOT_AVAILABLE);
}

CLDA::~CLDA()
{
}

bool CLDA::train_machine(CFeatures *data)
{
	REQUIRE(m_labels, "Labels for the given features are not specified!\n")
	REQUIRE(m_labels->get_label_type()==LT_BINARY, "The labels should of type"
			" CBinaryLabels! you provided %s \n",m_labels->get_name())

	if(data)
	{
		if(!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}

	REQUIRE(features, "Features are not provided!\n")
	SGVector<int32_t>train_labels=((CBinaryLabels *)m_labels)->get_int_labels();
	REQUIRE(train_labels.vector,"Provided Labels are empty!\n")

	SGMatrix<float64_t>feature_matrix=((CDenseFeatures<float64_t>*)features)
										->get_feature_matrix();
	int32_t num_feat=feature_matrix.num_rows;
	int32_t num_vec=feature_matrix.num_cols;
	REQUIRE(num_vec==train_labels.vlen,"Number of training examples(%d) should be "
		"equal to number of labels specified(%d)!\n", num_vec, train_labels.vlen);

	SGVector<int32_t> classidx_neg(num_vec);
	SGVector<int32_t> classidx_pos(num_vec);

	int32_t i=0;
	int32_t num_neg=0;
	int32_t num_pos=0;

	for(i=0; i<train_labels.vlen; i++)
	{
		if (train_labels.vector[i]==-1)
			classidx_neg[num_neg++]=i;

		else if(train_labels.vector[i]==+1)
			classidx_pos[num_pos++]=i;
	}

	w=SGVector<float64_t>(num_feat);
	w.zero();
	MatrixXd fmatrix=Map<MatrixXd>(feature_matrix.matrix, num_feat, num_vec);
	VectorXd mean_neg(num_feat);
	mean_neg=VectorXd::Zero(num_feat);
	VectorXd mean_pos(num_feat);
	mean_pos=VectorXd::Zero(num_feat);

	//mean neg
	for(i=0; i<num_neg; i++)
		mean_neg+=fmatrix.col(classidx_neg[i]);
	mean_neg/=(float64_t)num_neg;

	// get m(-ve) - mean(-ve)
	for(i=0; i<num_neg; i++)
		fmatrix.col(classidx_neg[i])-=mean_neg;

	//mean pos
	for(i=0; i<num_pos; i++)
		mean_pos+=fmatrix.col(classidx_pos[i]);
	mean_pos/=(float64_t)num_pos;

	// get m(+ve) - mean(+ve)
	for(i=0; i<num_pos; i++)
		fmatrix.col(classidx_pos[i])-=mean_pos;

	SGMatrix<float64_t>scatter_matrix(num_feat, num_feat);
	Map<MatrixXd> scatter(scatter_matrix.matrix, num_feat, num_feat);

	if (m_method == FLD_LDA || (m_method==AUTO_LDA && num_vec>num_feat))
	{
		// covariance matrix.
		MatrixXd cov_mat(num_feat, num_feat);
		cov_mat=fmatrix*fmatrix.transpose();
		scatter=cov_mat/(num_vec-1);
		float64_t trace=scatter.trace();
		double s=1.0-m_gamma;
		scatter *=s;
		scatter.diagonal()+=VectorXd::Constant(num_feat, trace*m_gamma/num_feat);

		// the usual way
		// we need to find a Basic Linear Solution of A.x=b for 'x'.
		// Instead of crudely Inverting A, we go for solve() using Decompositions.
		// where:
		// MatrixXd A=scatter;
		// VectorXd b=mean_pos-mean_neg;
		// VectorXd x=w;
		Map<VectorXd> x(w.vector, num_feat);
		LLT<MatrixXd> decomposition(scatter);
		x=decomposition.solve(mean_pos-mean_neg);

		// get the weights w_neg(for -ve class) and w_pos(for +ve class)
		VectorXd w_neg=decomposition.solve(mean_neg);
		VectorXd w_pos=decomposition.solve(mean_pos);

		// get the bias.
		bias=0.5*(w_neg.dot(mean_neg)-w_pos.dot(mean_pos));
	}

	else
	{
		//for algorithmic detail, please refer to section 16.3.1. of Bayesian
		//Reasoning and Machine Learning by David Barber.

		//we will perform SVD here.
		MatrixXd fmatrix1=Map<MatrixXd>(feature_matrix.matrix, num_feat, num_vec);

		// to hold the centered positive and negative class data
		MatrixXd cen_pos(num_feat,num_pos);
		MatrixXd cen_neg(num_feat,num_neg);

		for(i=0; i<num_pos;i++)
			cen_pos.col(i)=fmatrix.col(classidx_pos[i]);

		for(i=0; i<num_neg;i++)
			cen_neg.col(i)=fmatrix.col(classidx_neg[i]);

		//+ve covariance matrix
		cen_pos=cen_pos*cen_pos.transpose()/(float64_t(num_pos-1));

		//-ve covariance matrix
		cen_neg=cen_neg*cen_neg.transpose()/(float64_t(num_neg-1));

		//within class matrix
		MatrixXd Sw= num_pos*cen_pos+num_neg*cen_neg;
		float64_t trace=Sw.trace();
		double s=1.0-m_gamma;
		Sw *=s;
		Sw.diagonal()+=VectorXd::Constant(num_feat, trace*m_gamma/num_feat);

		//total mean
		VectorXd mean_total=(num_pos*mean_pos+num_neg*mean_neg)/(float64_t)num_vec;

		//between class matrix
		MatrixXd Sb(num_feat,2);
		Sb.col(0)=sqrt(num_pos)*(mean_pos-mean_total);
		Sb.col(1)=sqrt(num_neg)*(mean_neg-mean_total);

		JacobiSVD<MatrixXd> svd(fmatrix1, ComputeThinU);

		// basis to represent the solution
		MatrixXd Q=svd.matrixU();
		// modified between class scatter
		Sb=Q.transpose()*(Sb*(Sb.transpose()))*Q;

		// modified within class scatter
		Sw=Q.transpose()*Sw*Q;

		// to find SVD((inverse(Chol(Sw)))' * Sb * (inverse(Chol(Sw))))
		//1.get Cw=Chol(Sw)
		//find the decomposition of Cw'.
		HouseholderQR<MatrixXd> decomposition(Sw.llt().matrixU().transpose());

		//2.get P=inv(Cw')*Sb_new
		//MatrixXd P=decomposition.solve(Sb);
		//3. final value to be put in SVD will be therefore:
		// final_ output =(inv(Cw')*(P'))'
		JacobiSVD<MatrixXd> svd2(decomposition.solve((decomposition.solve(Sb))
					.transpose()).transpose(), ComputeThinU);

		// Since this is a linear classifier, with only binary classes,
		// we need to keep only the 1st eigenvector.
		Map<VectorXd> x(w.vector, num_feat);
		x=Q*(svd2.matrixU().col(0));
		// get the bias
		bias=(x.transpose()*mean_total);
		bias=bias*(-1);
	}
	return true;
}
#endif//HAVE_EIGEN3
