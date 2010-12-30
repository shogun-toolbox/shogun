/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010-2011 Alexander Binder
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010-2011 Berlin Institute of Technology
 */

#include "RandomFourierGaussPreproc.h"
#include <cmath>

using namespace shogun;

void CRandomFourierGaussPreproc::copy(const CRandomFourierGaussPreproc & feats) {

	dim_input_space = feats.dim_input_space;
	cur_dim_input_space = feats.cur_dim_input_space;
	dim_feature_space = feats.dim_feature_space;
	randomcoeff_additive = feats.randomcoeff_additive;
	kernelwidth=feats.kernelwidth;
	
	
	randomcoeff_multiplicative.resize(feats.randomcoeff_multiplicative.size());
	for(size_t i=0;i<randomcoeff_multiplicative.size() ;++i)
	{
		if(feats.randomcoeff_multiplicative[i]!=NULL)
		{
			randomcoeff_multiplicative[i]=new float64_t[cur_dim_input_space];
			std::copy(feats.randomcoeff_multiplicative[i],feats.randomcoeff_multiplicative[i]+cur_dim_input_space,randomcoeff_multiplicative[i]);
		}
		else
		{
			randomcoeff_multiplicative[i]=NULL;
		}
	}

}

CRandomFourierGaussPreproc::CRandomFourierGaussPreproc() :
	CSimplePreProc<float64_t> ("RandomFourierGaussPreproc", "RFGA") {
	dim_feature_space = 1000;
	dim_input_space = 0;
	cur_dim_input_space = 0;
	kernelwidth=1;

}

CRandomFourierGaussPreproc::CRandomFourierGaussPreproc(
		const CRandomFourierGaussPreproc & feats) :
	CSimplePreProc<float64_t> ("RandomFourierGaussPreproc", "RFGA") {
	copy(feats);
}

CRandomFourierGaussPreproc & CRandomFourierGaussPreproc::operator=(
		const CRandomFourierGaussPreproc & feats) {
	copy(feats);
	return (*this);
}

CRandomFourierGaussPreproc::~CRandomFourierGaussPreproc() {
	for(size_t i=0;i<randomcoeff_multiplicative.size() ;++i)
	{
		delete[] randomcoeff_multiplicative[i];
	}
}

EFeatureClass CRandomFourierGaussPreproc::get_feature_class() {
	return C_SIMPLE;
}

EFeatureType CRandomFourierGaussPreproc::get_feature_type() {
	return F_DREAL;
}

int32_t CRandomFourierGaussPreproc::get_dim_feature_space() const {
	return ((int32_t) dim_feature_space);
}

void CRandomFourierGaussPreproc::set_dim_feature_space(const int32_t dim) {
	if (dim <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_dim_feature_space(const int32 dim): dim<=0 is not allowed");
	}

	dim_feature_space = dim;

}

int32_t CRandomFourierGaussPreproc::get_dim_input_space() const {
	return ((int32_t) dim_input_space);
}

void CRandomFourierGaussPreproc::set_kernelwidth(const float64_t kernelwidth2 ) {
	if (kernelwidth2 <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_kernelwidth(const float64_t kernelwidth2 ): kernelwidth2 <= 0 is not allowed");
	}
	kernelwidth=kernelwidth2;
}

float64_t CRandomFourierGaussPreproc::get_kernelwidth( ) const {
	return (kernelwidth);
}

void CRandomFourierGaussPreproc::set_dim_input_space(const int32_t dim) {
	if (dim <= 0) {
		throw ShogunException(
				"void CRandomFourierGaussPreproc::set_dim_input_space(const int32 dim): dim<=0 is not allowed");
	}

	dim_input_space = dim;

}

bool CRandomFourierGaussPreproc::test_rfinited() const {

	if ((dim_feature_space == (int32_t) randomcoeff_additive.size())
			&& (dim_feature_space
					== (int32_t) randomcoeff_multiplicative.size())
			&& (dim_input_space > 0) && (dim_feature_space > 0)) {
		if (dim_input_space == cur_dim_input_space) {
			// already inited
			return true;
		} else {
			return false;
		}
	}

	return false;
}

bool CRandomFourierGaussPreproc::init_randomcoefficients() {
	if (dim_feature_space <= 0) {
		throw ShogunException(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients(): dim_feature_space<=0 is not allowed\n");
	}
	if (dim_input_space <= 0) {
		throw ShogunException(
				"bool CRandomFourierGaussPreproc::init_randomcoefficients(): dim_input_space<=0 is not allowed\n");
	}

	if (test_rfinited()) {
		return false;
	}


	SG_INFO("initializing randomcoefficients \n") ;

	float64_t pi = 3.14159265;
	
	for(size_t i=0;i<randomcoeff_multiplicative.size() ;++i)
	{
		delete[] randomcoeff_multiplicative[i];
	}
	randomcoeff_multiplicative.clear();

	randomcoeff_additive.resize(dim_feature_space);
	randomcoeff_multiplicative.resize(dim_feature_space);
	for (size_t i = 0; i < randomcoeff_additive.size(); ++i) {
		randomcoeff_additive[i] = CMath::random((float64_t) 0.0, 2 * pi);
	}
	for (size_t i = 0; i < randomcoeff_multiplicative.size(); ++i) {
		cur_dim_input_space = dim_input_space;
		randomcoeff_multiplicative[i] = new float64_t[cur_dim_input_space];
		for (int32_t k = 0; k < cur_dim_input_space; ++k) {
			float64_t x1,x2;
			float64_t s = 2;
			while ((s >= 1) ) {
				// Marsaglia polar for gaussian
				x1 = CMath::random((float64_t) -1.0, (float64_t) 1.0);
				x2 = CMath::random((float64_t) -1.0, (float64_t) 1.0);
				s=x1*x1+x2*x2;
			}

			// =  x1/CMath::sqrt(val)* CMath::sqrt(-2*CMath::log(val));
			randomcoeff_multiplicative[i][k] =  x1*CMath::sqrt(-2*CMath::log(s)/s )/kernelwidth;
		}
	}

	SG_INFO("finished: initializing randomcoefficients \n") ;

	return true;
}

void CRandomFourierGaussPreproc::get_randomcoefficients(
		float64_t ** randomcoeff_additive2,
		float64_t ** randomcoeff_multiplicative2, int32_t *dim_feature_space2,
		int32_t *dim_input_space2) const {

	ASSERT(randomcoeff_additive2);
	ASSERT(randomcoeff_multiplicative2);

	if (!test_rfinited()) {
		*dim_feature_space2 = 0;
		*dim_input_space2 = 0;
		*randomcoeff_additive2 = NULL;
		*randomcoeff_multiplicative2 = NULL;
		return;
	}

	*dim_feature_space2 = randomcoeff_additive.size();
	*dim_input_space2 = dim_input_space;

	*randomcoeff_additive2 = new float64_t[randomcoeff_additive.size()];
	*randomcoeff_multiplicative2 = new float64_t[randomcoeff_additive.size()
			* dim_input_space];

	std::copy(randomcoeff_additive.begin(), randomcoeff_additive.end(),
			*randomcoeff_additive2);
	for (size_t i = 0; i < randomcoeff_additive.size(); ++i) {
		std::copy(randomcoeff_multiplicative[i], randomcoeff_multiplicative[i]
				+ cur_dim_input_space, (*randomcoeff_multiplicative2) + i
				* dim_input_space);

	}

}

void CRandomFourierGaussPreproc::set_randomcoefficients(
		float64_t *randomcoeff_additive2,
		float64_t * randomcoeff_multiplicative2,
		const int32_t dim_feature_space2, const int32_t dim_input_space2) {
	dim_feature_space = dim_feature_space2;
	dim_input_space = dim_input_space2;
	
	for(size_t i=0;i<randomcoeff_multiplicative.size() ;++i)
	{
		delete[] randomcoeff_multiplicative[i];
	}
	randomcoeff_multiplicative.clear();

	randomcoeff_additive.resize(dim_feature_space);
	randomcoeff_multiplicative.resize(dim_feature_space);
	for (size_t i = 0; i < randomcoeff_multiplicative.size(); ++i) {
		std::copy(randomcoeff_additive2, randomcoeff_additive2
				+ dim_feature_space, randomcoeff_additive.begin());

		cur_dim_input_space = dim_input_space;
		randomcoeff_multiplicative[i] = new float64_t[cur_dim_input_space];
		std::copy(randomcoeff_multiplicative2 + i * dim_input_space,
				randomcoeff_multiplicative2 + (i + 1) * dim_input_space,
				randomcoeff_multiplicative[i]);

	}

}

bool CRandomFourierGaussPreproc::init(CFeatures *f) {
	if (f->get_feature_class() != get_feature_class()) {
		throw ShogunException(
				"CRandomFourierGaussPreproc::init (CFeatures *f) requires CSimpleFeatures<float64_t> as features\n");
	}
	if (f->get_feature_type() != get_feature_type()) {
		throw ShogunException(
				"CRandomFourierGaussPreproc::init (CFeatures *f) requires CSimpleFeatures<float64_t> as features\n");
	}
	if (dim_feature_space <= 0) {
		throw ShogunException(
				"CRandomFourierGaussPreproc::init (CFeatures *f): dim_feature_space<=0 is not allowed, use void set_dim_feature_space(const int32 dim) before!\n");
	}

	SG_INFO("calling CRandomFourierGaussPreproc::init(...)\n");
	int32_t num_features =
			((CSimpleFeatures<float64_t>*) f)->get_num_features();

	if (!test_rfinited()) {
		dim_input_space = num_features;
		init_randomcoefficients();
		ASSERT( test_rfinited());
		return true;
	} else {
		dim_input_space = num_features;
		// does not reinit if dimension is the same to avoid overriding a previous call of set_randomcoefficients(...)
		bool inited = init_randomcoefficients();
		return inited;
	}

}

float64_t * CRandomFourierGaussPreproc::apply_to_feature_vector(float64_t *f,
		int32_t &len) {
	if (!test_rfinited()) {
		throw ShogunException(
				"float64_t * CRandomFourierGaussPreproc::apply_to_feature_vector(...): test_rfinited()==false: you need to call before CRandomFourierGaussPreproc::init (CFeatures *f) OR 	1. set_dim_feature_space(const int32 dim), 2. set_dim_input_space(const int32 dim), 3. init_randomcoefficients() or set_randomcoefficients(...) \n");
	}

	float64_t val = CMath::sqrt(2.0 / dim_feature_space);
	len = dim_feature_space;
	float64_t *res = new float64_t[dim_feature_space];

	for (int32_t od = 0; od < dim_feature_space; ++od) {
		res[od] = val * cos(randomcoeff_additive[od] + CMath::dot(f,
				randomcoeff_multiplicative[od], dim_input_space));
	}

	return res;
}

float64_t * CRandomFourierGaussPreproc::apply_to_feature_matrix(CFeatures *f) {
	init(f);

	// version for case dim_feature_space < dim_input space with direct transformation on feature matrix ?? 
	
	int32_t num_vectors = 0;
	int32_t num_features = 0;
	float64_t* m = ((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(
			num_features, num_vectors);
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features);
	if (m) {
		float64_t* res = new float64_t[num_vectors * dim_feature_space];
		if (res == NULL) {
			throw ShogunException(
					"CRandomFourierGaussPreproc::apply_to_feature_matrix(...): memory allocation failed \n");
		}
		float64_t val = CMath::sqrt(2.0 / dim_feature_space);

		for (int32_t vec = 0; vec < num_vectors; vec++) {
			for (int32_t od = 0; od < dim_feature_space; ++od) {
				res[od + vec * dim_feature_space] = val * cos(
						randomcoeff_additive[od]
								+ CMath::dot(m+vec * num_features,
										randomcoeff_multiplicative[od],
										dim_input_space));
			}
		}
		((CSimpleFeatures<float64_t>*) f)->set_feature_matrix(res,
				dim_feature_space, num_vectors);
		
		m = ((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(
				num_features, num_vectors);
		ASSERT(num_features==dim_feature_space);

		return res;
	} else {
		return (NULL);
	}
}

void CRandomFourierGaussPreproc::cleanup()
{
	
}
