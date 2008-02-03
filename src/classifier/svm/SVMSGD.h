#ifndef _SVMOCAS_H___
#define _SVMOCAS_H___

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
		CSVMSGD(DREAL lambda);
		CSVMSGD(DREAL lambda, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);

		~CSVMSGD();

		virtual bool train();


		inline void set_lambda(DREAL l) { lambda=l; }

		inline DREAL get_lambda() { return lambda; }

		inline void set_epochs(INT e) { epochs=e; }

		inline INT get_epochs() { return epochs; }

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
		void calibrate();

	private:
		DREAL t;
		DREAL lambda;
		DREAL wscale;
		DREAL bscale;
		INT epochs;
		INT skip;
		INT count;

		bool use_bias;
		bool use_regularized_bias;
};
#endif
