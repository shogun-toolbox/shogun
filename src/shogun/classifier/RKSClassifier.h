/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _RKSCLASSIFIER_H__
#define _RKSCLASSIFIER_H__

#include <shogun/machine/RKSMachine.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibLinear.h>

namespace shogun
{

/** @brief class that implements the Random Kitchen Sinks
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
class CRKSClassifier : public CRKSMachine
{
public:

	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_BINARY);

	/** default constructor */
	CRKSClassifier();

	/** constructor 
	 * 
	 * @param dataset the dataset to work on
	 * @param num_samples the number of samples to draw from the probability distribution p
	 * @param (*phi) the function to use to transform the features. It should take two arguments,
	 * 	the first being a feature vector X and the second a parameter vector A that was drawn from p
	 * @param (*p) the probability distribution function to use when drawing the function parameters a.
	 *  It should expect a single int argument that tells it how many samples to create.
	 */
	CRKSClassifier(CFeatures* dataset, CLabels* labels, int32_t num_samples,
			float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	
			SGVector<float64_t> (*p)());

	/** constructor with user-defined random coefficients
	 * 
	 * @param dataset the dataset to work on
	 * @param (*phi) the function to use to transform the features. It should take two arguments,
	 * 	the first being a feature vector X and the second a parameter vector A that was drawn from p
	 * @param a the random parameters a to use with the function phi 
	 */
	CRKSClassifier(CFeatures* dataset, CLabels* labels,
			float64_t (*phi)(SGVector<float64_t>, SGVector<float64_t>),	
			SGMatrix<float64_t> a);

	/** destructor */
	virtual ~CRKSClassifier();

	/** train machine
	 *
	 * @param data training data 
	 * @return whether training was successful
	 */
	bool train(CFeatures* data = NULL);

	/** apply machine to data
	 * if data is not specified apply to the current features
	 *
	 * @param data (test)data to be classified
	 * @return classified labels
	 */
	CLabels* apply(CFeatures* feats=NULL);

	/** apply linear machine to data
	 * for binary classification problem
	 *
	 * @param data (test)data to be classified
	 * @return classified labels
	 */
	virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

	/** applies to one vector 
	 *
	 * @param vec_idx index of the feature vector
	 */
	virtual float64_t apply_one(int32_t vec_idx);

	/** @return object name */
	const char* get_name() const; 

	/** sets the C's for the linear classifier
	 *
	 * @param C the C to use
	 */
	void set_C(float64_t C);

	/** sets the epsilon to use for the linear classifier
	 *
	 * @param epsilon the epsilon
	 */
	void set_epsilon(float64_t epsilon);

private:
	void init();

private:

	/** the solver used */
	CLibLinear* linear_classifier;

	/** default epsilon to use in the linear svm */
	static const float64_t DEFAULT_EPSILON;

	/** default C to use in the linear svm */
	static const float64_t DEFAULT_C;
};
}
#endif // _RKSCLASSIFIER_H__
