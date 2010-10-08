/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Jonas Behr
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __TRAINPREDMASTER_h__
#define __TRAINPREDMASTER_h__

#include "lib/common.h"
#include "base/SGObject.h"
/*
#include "ui/GUIKernel.h"
#include "genomic_signals/KernelFactory.h"
*/

namespace shogun
{
class CGUIKernel;
class CKernelFactory;

/** @brief class TrainPredMaster */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CTrainPredMaster : public CSGObject 
{
	public:
		/** constructor
		 */
		CTrainPredMaster(void);

		/** constructor
		 */
		CTrainPredMaster(CGUIKernel* p_ui_kernel);

		virtual ~CTrainPredMaster();

		/**
		 *
		 */
		void read_models_from_file(char* filename);

		/** 
		 * @return object name 
		 */
		inline virtual const char* get_name() const { return "TrainPredMaster"; }
	protected:

		CGUIKernel* ui_kernel;
		CKernelFactory** kernelplist;
};
}
#endif
