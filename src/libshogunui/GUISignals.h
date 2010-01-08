/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */


#ifndef _GUISIGNALS_H__
#define _GUISIGNALS_H__ 

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/signals/TrainPredMaster.h>

class CSGInterface;

class CGUISignals : public CSGObject
{
	public:
		CGUISignals(CSGInterface* interface);
		~CGUISignals();

		/** @return object name */
		inline virtual const char* get_name() const { return "GUISignals"; }

	protected:
		CSGInterface* ui;
		CTrainPredMaster* m_tpm;
};
#endif

