/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _RKSMACHINE_H__
#define _RKSMACHINE_H__

#include <shogun/machine/Machine.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{

/** @brief class that implements the generic interface for Random Kitchen Sinks
 * as mentioned in http://books.nips.cc/papers/files/nips21/NIPS2008_0885.pdf.
 *
 * This class expects:
 * 		a dataset to work on
 * 		a function phi such that |phi(x; a)| <= 1, the a's are the function parameters
 *		a probability distrubution p, from which to draw the a's
 *		the number of samples K to draw from p.
 *
 * Then:
 *		it draws K a's from p
 *		it computes for each vector in the dataset 
 *			Zi = [phi(Xi;a0), ..., phi(Xi;aK)]
 *		and then solves the empirical risk minimization problem for all Zi, either
 *			through least squares or through a linear SVM.
 *
 * Further useful resources, include :
 * 	http://www.shloosl.com/~ali/random-features/
 * 	https://research.microsoft.com/apps/video/dl.aspx?id=103390&l=i
 */
class CRKSMachine : public CMachine
{
public:

	/** default constructor */
	CRKSMachine();

	/** constructor 
	 * 
	 * @param dataset the dataset to work on
	 * @param num_samples the number of samples to draw from the probability distribution p
	 * @param (*phi) the function to use to transform the features. It should take two arguments,
	 * 	the first being a feature vector X and the second a parameter vector A that was drawn from p
	 * @param (*p) the probability distribution function to use when drawing the function parameters a.
	 *  It should expect a single int argument that tells it how many samples to create.
	 */
	CRKSMachine(CFeatures* dataset, CLabels* labels, int32_t num_samples,
			float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	
			SGVector<float64_t> (*p)());

	/** constructor with user-defined random coefficients
	 * 
	 * @param dataset the dataset to work on
	 * @param (*phi) the function to use to transform the features. It should take two arguments,
	 * 	the first being a feature vector X and the second a parameter vector A that was drawn from p
	 * @param a the random parameters a to use with the function phi 
	 */
	CRKSMachine(CFeatures* dataset, CLabels* labels,
			float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	
			SGMatrix<float64_t> a);

	/** destructor */
	virtual ~CRKSMachine();

	/** samples #num_samples many coefficients for the function phi, from p
	 * returns a SGMatrix with as many columns as the number of samples and 
	 * as many rows as the number of parameters returned from p. 
	 *
	 * @param p the probability distribution from which to draw the parameters a
	 * @param num_samples how many times to draw parameters from p
	 * @return a SGMatrix containing the parameters that were drawn from p
	 */
	static SGMatrix<float64_t> generate_random_coefficients(
			SGVector<float64_t> (*p)(), int32_t num_samples);

	/** set the phi function which is used to transform the data.
	 * The function should expect two arguments as SGVector<float64_t>,
	 * where the first specifies the feature vector x and the second the function
	 * parameters a. 
	 *
	 * @param phi the function used to transform the data
	 */
	void set_phi_function(float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>));

	/** sets the function from which to draw the function parameters.
	 *
	 * @parameter p the function from which to generate the random function parameters
	 */
	void set_p_function(SGVector<float64_t> (*p)());

	/** converts the specified features to the dense representation described in the class
	 * description, using the function phi and the random function parameters that were
	 * generated through p.
	 *
	 * @param feats the features to convert
	 * @param phi to function to use in the conversion
	 * @param random_params the function phi parameters
	 * @return the converted features
	 */
	static CDenseFeatures<float64_t>* convert_data(CFeatures* feats,
				float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),
				SGMatrix<float64_t> random_params);

	/** returns the random function parameters that were generated through the function p
	 *
	 * @return the generated random coefficients
	 */
	SGMatrix<float64_t> get_random_coefficients();

	/** transforms the provided features using the random parameters that
	 * were generated or supplied and the function phi that must have been 
	 * provided.
	 *
	 * @param feats the DotFeatures to transform and set
	 */ 
	void set_features(CFeatures* feats);

	/** Returns the transformed features
	 *
	 * @return the transformed features
	 */
	CDenseFeatures<float64_t>* get_features() const;

	/** @return object name */
	const char* get_name() const; 

private:
	void init(CDenseFeatures<float64_t>* dataset, CLabels* labels, SGMatrix<float64_t> random_params);

protected:

	/** the dataset */
	CDenseFeatures<float64_t>* m_dataset;

private:

	/** random coefficients of the function phi, drawn from p */
	SGMatrix<float64_t> random_coeff;

	/** the phi function used to transform the data */
	float64_t (*m_phi)(SGVector<float64_t>, SGVector<float64_t>);
};
}

#endif // _RKSMACHINE_H__
