/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LDA_H___
#define _LDA_H___

#include "lib/common.h"

#ifdef HAVE_LAPACK
#include "features/Features.h"
#include "classifier/LinearClassifier.h"

class CLDA : public CLinearClassifier
{
	public:
		CLDA(DREAL prior=1);
		virtual ~CLDA();

		virtual bool train();

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LDA;
		}
	protected:
		DREAL prior;
};
#endif
#endif
