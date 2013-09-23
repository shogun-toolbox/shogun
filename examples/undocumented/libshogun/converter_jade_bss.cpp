/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 *
 * Thanks to Andreas Ziehe
 */

#include <shogun/base/init.h>
#include <shogun/lib/common.h>

#include <iostream>

using namespace shogun;

#ifdef HAVE_EIGEN3

#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

#include <shogun/converter/ica/Jade.h>
#include <shogun/evaluation/ica/PermutationMatrix.h>
#include <shogun/evaluation/ica/AmariIndex.h>

using namespace Eigen;

void test()
{
	// Generate sample data	
	CMath::init_random(0); 
	int n_samples = 2000;
	VectorXd time(n_samples, true);
	time.setLinSpaced(n_samples,0,10);

	// Source Signals
	MatrixXd S(2,n_samples);
	for(int i = 0; i < n_samples; i++)
	{
		// Sin wave
		S(0,i) = sin(2*time[0,i]);
		S(0,i) += 0.2*CMath::randn_double(); 
		
		// Square wave
		S(1,i) = sin(3*time[0,i]) < 0 ? -1 : 1;
		S(1,i) += 0.2*CMath::randn_double();
	}
	
	// Standardize data
	VectorXd avg = S.rowwise().sum() / n_samples;
	VectorXd std = ((S.colwise() - avg).array().pow(2).rowwise().sum() / n_samples).array().sqrt();
	for(int i = 0; i < n_samples; i++)
		S.col(i) = S.col(i).cwiseQuotient(std);

	// Mixing Matrix
	SGMatrix<float64_t> mixing_matrix(2,2);
	Map<MatrixXd> A(mixing_matrix.matrix,2,2);
	A(0,0) = 1;    A(0,1) = 0.5;
	A(1,0) = 0.5;  A(1,1) = 1;

	std::cout << "Mixing Matrix:" << std::endl;
	std::cout << A << std::endl << std::endl;

	// Mix signals
	SGMatrix<float64_t> X(2,n_samples);
	Map<MatrixXd> EX(X.matrix,2,n_samples);
	EX = A * S;
	CDenseFeatures< float64_t >* mixed_signals = new CDenseFeatures< float64_t >(X);

	// Separate
	CJade* jade = new CJade();
	SG_REF(jade);

	CFeatures* signals = jade->apply(mixed_signals);
	SG_REF(signals);

	// Close to a permutation matrix (with random scales)
	Map<MatrixXd> EA(jade->get_mixing_matrix().matrix,2,2);
	
	std::cout << "Estimated Mixing Matrix:" << std::endl;
	std::cout << EA << std::endl << std::endl;
	
	SGMatrix<float64_t> P(2,2);
	Eigen::Map<MatrixXd> EP(P.matrix,2,2);
	EP = EA.inverse() * A;

	bool isperm = is_permutation_matrix(P);
	std::cout << "EA^-1 * A == Permuatation Matrix is: " << isperm << std::endl;

	float64_t amari_err = amari_index(jade->get_mixing_matrix(), mixing_matrix, true); 
	std::cout << "Amari Error: " << amari_err << std::endl;

	SG_UNREF(jade);
	SG_UNREF(mixed_signals);
	SG_UNREF(signals);
	
	return;
}

#endif // HAVE_EIGEN3

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

#ifdef HAVE_EIGEN3
	test();
#endif // HAVE_EIGEN3
	exit_shogun();

	return 0;
}