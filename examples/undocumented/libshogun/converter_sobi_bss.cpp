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
#include <shogun/features/DenseFeatures.h>

#include <iostream>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

#include <shogun/converter/ica/SOBI.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;
typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;

using namespace shogun;

void test()
{
	// Generate sample data	
	CMath::init_random(0); 
	int n_samples = 2000;
	EVector time(n_samples, true);
	time.setLinSpaced(n_samples,0,10);

	// Source Signals
	EMatrix S(2,n_samples);
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
	EVector avg = S.rowwise().sum() / n_samples;
	EVector std = ((S.colwise() - avg).array().pow(2).rowwise().sum() / n_samples).array().sqrt();
	for(int i = 0; i < n_samples; i++)
		S.col(i) = S.col(i).cwiseQuotient(std);

	// Mixing Matrix
	EMatrix A(2,2);
	A(0,0) = 1;    A(0,1) = 0.5;
	A(1,0) = 0.5;  A(1,1) = 1;

	std::cout << "Mixing Matrix:" << std::endl;
	std::cout << A << std::endl << std::endl;

	// Mix signals
	SGMatrix<float64_t> X(2,n_samples);
	Eigen::Map<EMatrix> EX(X.matrix,2,n_samples);
	EX = A * S;
	CDenseFeatures< float64_t >* mixed_signals = new CDenseFeatures< float64_t >(X);

	// Separate
	CSOBI* sobi = new CSOBI();
	SG_REF(sobi);

	CFeatures* signals = sobi->apply(mixed_signals);
	SG_REF(signals);

	// Estimated Mixing Matrix	
	SGMatrix<float64_t> est_A = sobi->get_mixing_matrix();
	Eigen::Map<EMatrix> est_EA(est_A.matrix, 2,2);
		
	std::cout << "Estimated Mixing Matrix:" << std::endl;
	std::cout << est_EA << std::endl << std::endl;

	// Separation error
	Eigen::Map<EMatrix> ES (((CDenseFeatures<float64_t>*)signals)->get_feature_matrix().matrix,2,n_samples);	
	double sep_error = (S-ES).array().abs().sum();

	std::cout << "Separation error: " << sep_error << std::endl;

	SG_UNREF(sobi);
	SG_UNREF(mixed_signals);
	SG_UNREF(signals);
	
	return;
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

#endif //HAVE_EIGEN3
