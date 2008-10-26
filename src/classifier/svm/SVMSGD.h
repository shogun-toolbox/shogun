#ifndef _SVMSGD_H___
#define _SVMSGD_H___

/*
   SVM with stochastic gradient
   Copyright (C) 2007- Leon Bottou

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

   Shogun adjustments (w) 2008 Soeren Sonnenburg
*/

#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

/** class SVMSGD */
class CSVMSGD : public CSparseLinearClassifier
{
	public:
		/** constructor
		 *
		 * @param C constant C
		 */
		CSVMSGD(DREAL C);

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CSVMSGD(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);

		~CSVMSGD();

		/** get classifier type
		 *
		 * @return classifier type SVMOCAS
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SVMSGD; }

		/** train SVM
		 *
		 * @return true if training was successful
		 */
		virtual bool train();

		/** set C
		 *
		 * @param c1 new C1
		 * @param c2 new C2
		 */
		inline void set_C(DREAL c1, DREAL c2) { C1=c1; C2=c2; }

		/** get C1
		 *
		 * @return C1
		 */
		inline DREAL get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline DREAL get_C2() { return C2; }

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

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** check if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set if regularized bias shall be enabled
		 *
		 * @param enable_bias if regularized bias shall be enabled
		 */
		inline void set_regularized_bias_enabled(bool enable_bias) { use_regularized_bias=enable_bias; }

		/** check if regularized bias is enabled
		 *
		 * @return if regularized bias is enabled
		 */
		inline bool get_regularized_bias_enabled() { return use_regularized_bias; }

	protected:
		/** calibrate */
		void calibrate();

	private:
		DREAL t;
		DREAL C1;
		DREAL C2;
		DREAL wscale;
		DREAL bscale;
		int32_t epochs;
		int32_t skip;
		int32_t count;

		bool use_bias;
		bool use_regularized_bias;
};
#endif
