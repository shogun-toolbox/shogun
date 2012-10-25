/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Hidekazu Oiwa
 */

#ifndef _AVERAGEDPERCEPTRON_H___
#define _AVERAGEDPERCEPTRON_H___

#include <stdio.h>
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
/** @brief Class Averaged Perceptron implements
 *         the standard linear (online) algorithm.
 *         Averaged perceptron is the simple extension of Perceptron.
 *
 * Given a maximum number of iterations (the standard averaged perceptron
 * algorithm is not guaranteed to converge) and a fixed lerning rate,
 * the result is a linear classifier.
 *
 * \sa CLinearMachine
 */
class CAveragedPerceptron : public CLinearMachine
{
	public:
		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor */
		CAveragedPerceptron();

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CAveragedPerceptron(CDotFeatures* traindat, CLabels* trainlab);
		virtual ~CAveragedPerceptron();

		/** get classifier type
		 *
		 * @return classifier type AVERAGEDPERCEPTRON
		 */
		virtual EMachineType get_classifier_type() { return CT_AVERAGEDPERCEPTRON; }

		/// set learn rate of gradient descent training algorithm
		inline void set_learn_rate(float64_t r)
		{
			learn_rate=r;
		}

		/// set maximum number of iterations
		inline void set_max_iter(int32_t i)
		{
			max_iter=i;
		}

		/** @return object name */
		virtual const char* get_name() const { return "AveragedPerceptron"; }

protected:

		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:
		/** learning rate */
		float64_t learn_rate;
		/** maximum number of iterations */
		int32_t max_iter;
};
}
#endif
