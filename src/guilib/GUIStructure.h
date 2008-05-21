/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */


#ifndef _GUISTRUCTURE_H__
#define _GUISTRUCTURE_H__ 

#include "lib/config.h"

#ifndef HAVE_SWIG

#include "base/SGObject.h"

#include "structure/Plif.h"
#include "structure/PlifArray.h"
#include "structure/PlifBase.h"
#include "structure/DynProg.h"

class CSGInterface;

class CGUIStructure : public CSGObject
{
	public:
		CGUIStructure(CSGInterface* interface);
		~CGUIStructure();

		bool set_plif_struct(INT N, INT M, DREAL* all_limits,
				DREAL* all_penalties, INT* ids, T_STRING<CHAR>* names,
				DREAL* min_values, DREAL* max_values, bool* all_use_cache,
				bool* all_use_svm, T_STRING<CHAR>* all_transform);

		inline CPlif** get_PEN()
		{
			return m_PEN;
		}
		inline INT get_num_plifs()
		{
			return m_N;
		}
		inline INT get_num_limits()
		{
			return m_M;
		}
	protected:
		CSGInterface* ui;
		CPlif** m_PEN;
		INT m_N;
		INT m_M;
};
#endif //HAVE_SWIG
#endif

