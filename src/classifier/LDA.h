/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LDA_H___
#define _LDA_H___

#include "lib/common.h"

#ifdef HAVE_LAPACK
#include "features/Features.h"
#include "classifier/LinearClassifier.h"

/** class LDA */
class CLDA : public CLinearClassifier
{
	public:
		/** constructor
		 *
		 * @param gamma gamma
		 */
		CLDA(DREAL gamma=0);

		/** constructor
		 *
		 * @param gamma gamma
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CLDA(DREAL gamma, CRealFeatures* traindat, CLabels* trainlab);
		virtual ~CLDA();

		/** set gamme
		 *
		 * @param gamma the new gamma
		 */
		inline void set_gamma(DREAL gamma)
		{
			m_gamma=gamma;
		}

		/** get gamma
		 *
		 * @return gamma
		 */
		inline DREAL get_gamma()
		{
			return m_gamma;
		}

		/** train  classifier
		 *
		 * @return if training was successful
		 */
		virtual bool train();

		/** get classifier type
		 *
		 * @return classifier type LDA
		 */
		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LDA;
		}

	protected:
		/** gamma */
		DREAL m_gamma;
};
#endif
#endif
