/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBLINEAR_H___
#define _LIBLINEAR_H___

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "lib/common.h"
#include "classifier/SparseLinearClassifier.h"

enum LIBLINEAR_LOSS
{
	LR = 0,
	L2 = 1
};

/** class to implement LibLinear */
class CLibLinear : public CSparseLinearClassifier
{
	public:
		/** constructor
		 *
		 * @param loss loss
		 */
		CLibLinear(LIBLINEAR_LOSS loss);

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab training labels
		 */
		CLibLinear(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab);

		virtual ~CLibLinear();

		/** train SVM */
		virtual bool train();

		/** get classifier type
		 *
		 * @return the classifier type
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_LIBLINEAR; }

		/** set C
		 *
		 * @param c1 C1
		 * @param c2 C2
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

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(DREAL eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline DREAL get_epsilon() { return epsilon; }

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

	protected:
		/** C1 */
		DREAL C1;
		/** C2 */
		DREAL C2;
		/** if bias shall be used */
		bool use_bias;
		/** epsilon */
		DREAL epsilon;

		/** loss */
		LIBLINEAR_LOSS loss;
};
#endif //HAVE_LAPACK
#endif //_LIBLINEAR_H___
