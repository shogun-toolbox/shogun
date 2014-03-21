#ifndef _SGDQN_H___
#define _SGDQN_H___

/*
   SVM with Quasi-Newton stochastic gradient
   Copyright (C) 2009- Antoine Bordes

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

   Shogun adjustments (w) 2011 Siddharth Kherada
*/

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/loss/LossFunction.h>

namespace shogun
{
/** @brief class SGDQN */
class CSGDQN : public CLinearMachine
{
	public:

		/** problem type */
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** default constructor  */
		CSGDQN();

		/** constructor
		 *
		 * @param C constant C
		 */
		CSGDQN(float64_t C);

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CSGDQN(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);

		virtual ~CSGDQN();

		/** get classifier type
		 *
		 * @return classifier type SVMSGDQN
		 */
		virtual EMachineType get_classifier_type() { return CT_SGDQN; }

		/** train classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** set C
		 *
		 * @param c_neg new C constant for negatively labeled examples
		 * @param c_pos new C constant for positively labeled examples
		 *
		 */
		inline void set_C(float64_t c_neg, float64_t c_pos) { C1=c_neg; C2=c_pos; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set epochs
		 *
		 * @param e new number of training epochs
		 */
		inline void set_epochs(int32_t e) { epochs=e; }

		/** get epochs
		 *
		 * @return the number of training epochs
		 */
		inline int32_t get_epochs() { return epochs; }

		/**computing diagonal scaling matrix B as ratio*/
		void compute_ratio(float64_t* W,float64_t* W_1,float64_t* B,float64_t* dst,int32_t dim,float64_t regularizer_lambda,float64_t loss);

		/** combine and clip scaling matrix B */
		void combine_and_clip(float64_t* Bc,float64_t* B,int32_t dim,float64_t c1,float64_t c2,float64_t v1,float64_t v2);

		/** Set the loss function to use
		 *
		 * @param loss_func object derived from CLossFunction
		 */
		void set_loss_function(CLossFunction* loss_func);

		/** Return the loss function
		 *
		 * @return loss function as CLossFunction*
		 */
		inline CLossFunction* get_loss_function() { SG_REF(loss); return loss; }

		/** @return object name */
		virtual const char* get_name() const { return "SGDQN"; }

	protected:
		/** calibrate */
		void calibrate();

	private:
		void init();

	private:
		float64_t t;
		float64_t C1;
		float64_t C2;
		int32_t epochs;
		int32_t skip;
		int32_t count;

		CLossFunction* loss;
};
}
#endif
