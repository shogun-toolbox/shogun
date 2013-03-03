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

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/DensePreprocessor.h>



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
class CRandomFourierGaussPreproc: public CPreprocessor {
public:
	/** default constructor */
	CRandomFourierGaussPreproc();

	/** alternative constructor */
	CRandomFourierGaussPreproc(const CRandomFourierGaussPreproc & pr);

	/** default destructor
	 * takes care for float64_t* randomcoeff_additive,float64_t* randomcoeff_multiplicative;
	 */
	~CRandomFourierGaussPreproc();


	/** default processing routine, 
	 * @param features the features to be processed, must be of type 
	 
	 * (features->get_feature_class()==C_SPARSE)
         * ||(features->get_feature_class()==C_DENSE)
	 * ||(features->get_feature_class()==C_BINNED_DOT)
	 * ||(features->get_feature_class()==C_COMBINED_DOT)
	 *
	 *   (features->get_feature_type()==F_SHORTREAL)
	 * ||(features->get_feature_type()==F_DREAL)
	 * ||(features->get_feature_type()==F_LONGREAL)
	 * ||(features->get_feature_type()==F_INT)
	 * ||(features->get_feature_type()==F_UINT)
	 * ||(features->get_feature_type()==F_LONG)
	 * ||(features->get_feature_type()==F_ULONG)
	 *
	 * uses dense dot
	 *
	 * USAGE WARNING: during testing you must use the same random coefficients as have been used during training, use get_randomcoefficients and set_randomcoefficients for securing that!
	 * always required is to call set_parameters() before usage
	 * a call to init_randomcoefficients_from_scratch(); resets coefficients (or sets them for the first time, set_parameters() does NOT set them)!
	 * 
	 * @return the processed feature matrix from the CDenseFeatures<float64_t> class
	 * in case (2) (see description above) this routine requires only steps 2a) and 2b), the rest is determined automatically
	 */
	CDenseFeatures<float64_t>* apply_to_dotfeatures_sparse_or_dense_with_real(CDotFeatures* features);


	/** inherited from base class
	 * @return C_DENSE
	 */
	virtual EFeatureType get_feature_type();

	/** inherited from base class
	 * @return F_DREAL
	 */
	virtual EFeatureClass get_feature_class();


	/** initializer routine
	 * sets parameters
	 * @param dim_input_space2 the dimensionality of the input features, given by the features you want to use
	 * @param dim_feature_space2 the dimensionality of the output features, a higher values gives a better approximation of the gaussian kernel, typically dim_feature_space2 >> dim_input_space2
	 * @param kernelwidth2 the gaussian kernel width, too small values result in bad approximations up to negative eigenvalues
	 * @return true always
	 */
        virtual bool set_parameters(const int32_t dim_input_space2, const int32_t dim_feature_space2, const float64_t kernelwidth2);


	/** computes new random coefficients 
	 * @return true always
	 */
	bool init_randomcoefficients_from_scratch();

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
	 * sets values of internal members 	randomcoeff_additive
	 * and randomcoeff_multiplicative
	 * simply use as input what you got from get_random_coefficients(...)
	 */
	void set_randomcoefficients(float64_t *randomcoeff_additive2,
			float64_t * randomcoeff_multiplicative2,
			const int32_t dim_feature_space2, const int32_t dim_input_space2, const float64_t kernelwidth2);




	/** a getter
	 * @return the set value of protected member dim_input_space
	 */
	int32_t get_dim_input_space() const;

	/** a getter
	 * @return the set value of protected member dim_feature_space
	 */
	int32_t get_dim_feature_space() const;

	/**  getter for kernel width
	 * @return kernel width
	 * throws exception if kernelwidth <=0
	 */
	float64_t get_kernelwidth( ) const;

	/** inherited from base class
	 * does nothing
	 */
	void cleanup();

	/// return the name of the preprocessor
	virtual const char* get_name() const { return "RandomFourierGaussPreproc"; }

	/// return a type of preprocessor
	virtual EPreprocessorType get_type() const { return P_RANDOMFOURIERGAUSS; }

	virtual bool init (CFeatures *features); // does nothing

protected:




	/**
	* checks whether this feature can be used woth RF preprocessor
	*/
	virtual bool check_applicability_to_feature(CFeatures* features);

	/**
	 * helper for copy constructor and assignment operator=
	 */
	void copy(const CRandomFourierGaussPreproc & feats); // helper for two constructors


	/** dimension of input features
	 * width of gaussian kernel in the form of exp(-x^2 / (2.0 kernelwidth^2) ) NOTE the 2.0 and the power ^2 !
	 */
	float64_t kernelwidth;


	/** desired dimension of input features as set by void set_dim_input_space(const int32_t dim)
	 *
	 */
	int32_t dim_input_space;


	/** desired dimension of output features  as set by void set_dim_feature_space(const int32_t dim)
	 *
	 */
	int32_t dim_feature_space;



	/**
	 * tests whether rf features have already been initialized
	 */
	bool test_rfinited() const;

	/**
	 * random coefficient
	 * length = cur_dim_feature_space
	 */
	float64_t* randomcoeff_additive;

	//CDenseFeatures<float64_t> randomcoeff_additive2; 

	/**
	 * random coefficient
	 * length = cur_dim_feature_space* cur_dim_input_space
	 */
	float64_t* randomcoeff_multiplicative;

	//CDenseFeatures<float64_t> randomcoeff_multiplicative2;
};
}
#endif
