/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Sergey Lisitsyn, 
 *          Heiko Strathmann, Saurabh Goyal
 */

#ifndef _PERCEPTRON_H___
#define _PERCEPTRON_H___

#include <shogun/lib/config.h>

#include <shogun/features/DotFeatures.h>
#include <shogun/lib/common.h>
#include <shogun/machine/IterativeLinearMachine.h>

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
class CPerceptron : public CIterativeLinearMachine
{
	public:

		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor */
		CPerceptron();

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

		/// set if the hyperplane should be initialized
		void set_initialize_hyperplane(bool initialize_hyperplane);

		/// get if the hyperplane should be initialized
		bool get_initialize_hyperplane();

		/** @return object name */
		virtual const char* get_name() const { return "Perceptron"; }

	protected:
		/** registers and initializes parameters */
		void init();
		/** init the model before training begins */
		virtual void init_model(CFeatures* data);
		/** perform a single iteration over features and update weights */
		virtual void iteration();

	protected:
		/** learning rate */
		float64_t learn_rate;

	private:
		/** whether the hyperplane should be initialized in train_machine
		 *
		 * this allows to initialize the hyperplane to externally set
		 * weights and bias
		 */
		bool m_initialize_hyperplane;
};
}
#endif
