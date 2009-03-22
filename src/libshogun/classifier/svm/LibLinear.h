/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIBLINEAR_H___
#define _LIBLINEAR_H___

#include "lib/config.h"

#ifdef HAVE_LAPACK
#include "lib/common.h"
#include "classifier/LinearClassifier.h"

enum LIBLINEAR_LOSS
{
	LR = 0,
	L2 = 1
};

/** @brief class to implement LibLinear */
class CLibLinear : public CLinearClassifier
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
		CLibLinear(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);

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
		inline void set_C(float64_t c1, float64_t c2) { C1=c1; C2=c2; }

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

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

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

		/** @return object name */
		inline virtual const char* get_name() const { return "LibLinear"; }

	protected:
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** if bias shall be used */
		bool use_bias;
		/** epsilon */
		float64_t epsilon;

		/** loss */
		LIBLINEAR_LOSS loss;
};
#endif //HAVE_LAPACK
#endif //_LIBLINEAR_H___
