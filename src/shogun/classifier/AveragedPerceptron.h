/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Fernando Iglesias, 
 *          Saurabh Goyal
 */

#ifndef _AVERAGEDPERCEPTRON_H___
#define _AVERAGEDPERCEPTRON_H___

#include <shogun/lib/config.h>

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
 * algorithm is not guaranteed to converge) and a fixed learning rate,
 * the result is a linear classifier.
 *
 * \sa CLinearMachine
 */
class SHOGUN_EXPORT CAveragedPerceptron : public CLinearMachine
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
		/** registers and initializes parameters */
		void init();

		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with training data)
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
