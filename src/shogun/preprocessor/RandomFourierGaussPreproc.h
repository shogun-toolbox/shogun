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

#ifndef _RANDOMFOURIERGAUSSPREPROC__H__
#define _RANDOMFOURIERGAUSSPREPROC__H__

#include <vector>
#include <algorithm>

#include <lib/common.h>
#include <mathematics/Math.h>
#include <preprocessor/DensePreprocessor.h>

namespace shogun {
/** @brief Preprocessor CRandomFourierGaussPreproc
 * implements Random Fourier Features for the Gauss kernel a la Ali Rahimi and Ben Recht Nips2007
 * after preprocessing the features using them in a linear kernel approximates a gaussian kernel
 *
 * approximation quality depends on dimension of feature space, NOT on number of data.
 *
 * effectively it requires two parameters for initialization: (A) the dimension of the input features stored in
 * dim_input_space
 * (B) the dimension of the output feature space
 *
 * in order to make it work there are two ways:
 * (1) if you have already previously computed random fourier features which you want to use together with
 * newly computed ones, then you have to take the random coefficients from the previous computation and provide them
 * via void set_randomcoefficients(...) for the new computation
 * this case is important for example if you compute separately features on training and testing data in two feature objects
 *
 * in this case you have to set
 * 1a) void set_randomcoefficients(...)
 *
 * (2) if you compute random fourier features from scratch
 * in this case you have to set
 * 2a) set_kernelwidth(...)
 * 2b) void set_dim_feature_space(const int32_t dim);
 * 2c) set_dim_input_space(const int32_t dim);
 * 2d) init_randomcoefficients() or apply_to_feature_matrix(...)
 */
class CRandomFourierGaussPreproc: public CDensePreprocessor<float64_t> {
public:
	/** default constructor */
	CRandomFourierGaussPreproc();

	/** alternative constructor */
	CRandomFourierGaussPreproc(const CRandomFourierGaussPreproc & pr);

	/** default destructor
	 * takes care for float64_t* randomcoeff_additive,float64_t* randomcoeff_multiplicative;
	 */
	~CRandomFourierGaussPreproc();

	/** default processing routine, inherited from base class
	 * @param features the features to be processed, must be of type CDenseFeatures<float64_t>
	 * @return the processed feature matrix from the CDenseFeatures<float64_t> class
	 * in case (2) (see description above) this routine requires only steps 2a) and 2b), the rest is determined automatically
	 */
	virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features); // ref count fo the feature matrix???


	/** alternative processing routine, inherited from base class
	 * @param vector the feature vector to be processed
	 * @return processed feature vector
	 * in order to work this routine requires the steps described above under cases (1) or two (2) before calling this routine
	 */
	virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

	/** inherited from base class
	 * @return C_DENSE
	 */
	virtual EFeatureType get_feature_type();

	/** inherited from base class
	 * @return F_DREAL
	 */
	virtual EFeatureClass get_feature_class();

	/** initializer routine
	 * calls set_dim_input_space(const int32_t dim); with the proper value
	 * calls init_randomcoefficients(); this call does NOT override a previous call to void set_randomcoefficients(...) IF and ONLY IF
	 * the dimensions of input AND feature space are equal to the values from the previous call to void set_randomcoefficients(...)
	 * @param f the features to be processed, must be of type CDenseFeatures<float64_t>
	 * @return true if new random coefficients were generated, false if old ones from a call to set_randomcoefficients(...) are kept
	 */
	virtual bool init(CFeatures *f);

	/**  setter for kernel width
	 * @param width kernel width to be set
	 */
	void set_kernelwidth(const float64_t width);

	/**  getter for kernel width
	 * @return kernel width
	 * throws exception if kernelwidth <=0
	 */
	float64_t get_kernelwidth( ) const;

	/**  getter for the random coefficients
	 * necessary for creating random fourier features compatible to the current ones
	 * returns values of internal members randomcoeff_additive
	 * and randomcoeff_multiplicative
	 */
	void get_randomcoefficients(float64_t ** randomcoeff_additive2,
			float64_t ** randomcoeff_multiplicative2,
			int32_t *dim_feature_space2, int32_t *dim_input_space2, float64_t* kernelwidth2 ) const;

	/**  setter for the random coefficients
	 * necessary for creating random fourier features compatible to the previous ones
	 * sets values of internal members	randomcoeff_additive
	 * and randomcoeff_multiplicative
	 * simply use as input what you got from get_random_coefficients(...)
	 */
	void set_randomcoefficients(float64_t *randomcoeff_additive2,
			float64_t * randomcoeff_multiplicative2,
			const int32_t dim_feature_space2, const int32_t dim_input_space2, const float64_t kernelwidth2);

	/** a setter
	 * @param dim the value of protected member dim_input_space
	 * throws a shogun exception if dim<=0
	 */
	void set_dim_input_space(const int32_t dim);

	/** a setter
	 * @param dim the value of protected member dim_feature_space
	 * throws a shogun exception if dim<=0
	 *
	 */
	void set_dim_feature_space(const int32_t dim);

	/** computes new random coefficients IF test_rfinited() evaluates to false
	 * test_rfinited() evaluates to TRUE if void set_randomcoefficients(...) hase been called and the values set by set_dim_input_space(...) , set_dim_feature_space(...) and set_kernelwidth(...) are consistent to the call of void set_randomcoefficients(...)
	 *
	 * throws shogun exception if dim_feature_space <= 0 or dim_input_space <= 0
	 *
	 * @return returns true if test_rfinited() evaluates to false and new coefficients are computed
	 * returns false if test_rfinited() evaluates to true and old random coefficients are kept which were set by a previous call to void set_randomcoefficients(...)
	 *
	 * this function is useful if you want to use apply_to_feature_vector but cannot call before it init(CFeatures *f)
	 *
	 */
	bool init_randomcoefficients();


	/** a getter
	 * @return the set value of protected member dim_input_space
	 */
	int32_t get_dim_input_space() const;

	/** a getter
	 * @return the set value of protected member dim_feature_space
	 */
	int32_t get_dim_feature_space() const;

	/** inherited from base class
	 * does nothing
	 */
	void cleanup();

	/// return the name of the preprocessor
	virtual const char* get_name() const { return "RandomFourierGaussPreproc"; }

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const { return P_RANDOMFOURIERGAUSS; }

protected:

	/**
	 * helper for copy constructor and assignment operator=
	 */
	void copy(const CRandomFourierGaussPreproc & feats); // helper for two constructors


	/** dimension of input features
	 * width of gaussian kernel in the form of exp(-x^2 / (2.0 kernelwidth^2) ) NOTE the 2.0 and the power ^2 !
	 */
	float64_t kernelwidth;

	/** dimension of input features
	 * width of gaussian kernel in the form of exp(-x^2 / (2.0 kernelwidth^2) ) NOTE the 2.0 and the power ^2 !
	 */
	float64_t cur_kernelwidth;

	/** desired dimension of input features as set by void set_dim_input_space(const int32_t dim)
	 *
	 */
	int32_t dim_input_space;

	/** actual dimension of input features as set by bool init_randomcoefficients() or void set_randomcoefficients
	 *
	 */
	int32_t cur_dim_input_space;


	/** desired dimension of output features  as set by void set_dim_feature_space(const int32_t dim)
	 *
	 */
	int32_t dim_feature_space;

	/** actual dimension of output features as set by bool init_randomcoefficients() or void set_randomcoefficients
	 *
	 */
	int32_t cur_dim_feature_space;

	/**
	 * tests whether rf features have already been initialized
	 */
	bool test_rfinited() const;

	/**
	 * random coefficient
	 * length = cur_dim_feature_space
	 */
	float64_t* randomcoeff_additive;

	/**
	 * random coefficient
	 * length = cur_dim_feature_space* cur_dim_input_space
	 */
	float64_t* randomcoeff_multiplicative;
};
}
#endif
