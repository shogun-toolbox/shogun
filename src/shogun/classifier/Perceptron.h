/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Sergey Lisitsyn, 
 *          Heiko Strathmann, Saurabh Goyal
 */

#ifndef _PERCEPTRON_H___
#define _PERCEPTRON_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
/** @brief Class Perceptron implements the standard linear (online) perceptron.
 *
 * Given a maximum number of iterations (the standard perceptron algorithm is
 * not guaranteed to converge) and a fixed lerning rate, the result is a linear
 * classifier.
 *
 * \sa CLinearMachine
 * \sa http://en.wikipedia.org/wiki/Perceptron
 */
class CPerceptron : public CLinearMachine
{
	public:

		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor */
		CPerceptron();

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CPerceptron(CDotFeatures* traindat, CLabels* trainlab);
		virtual ~CPerceptron();

		/** get classifier type
		 *
		 * @return classifier type PERCEPTRON
		 */
		virtual EMachineType get_classifier_type() { return CT_PERCEPTRON; }

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

		/// set if the hyperplane should be initialized
		void set_initialize_hyperplane(bool initialize_hyperplane);

		/// get if the hyperplane should be initialized
		bool get_initialize_hyperplane();

		/** @return object name */
		virtual const char* get_name() const { return "Perceptron"; }

	protected:
		/** registers and initializes parameters */
			void init();

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

	private:
		/** whether the hyperplane should be initialized in train_machine
		 *
		 * this allows to initialize the hyperplane externally using set_w and set_b
		 */
		bool m_initialize_hyperplane;
};
}
#endif
