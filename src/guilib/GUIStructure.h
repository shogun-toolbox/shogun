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
				INT* all_use_svm, T_STRING<CHAR>* all_transform);

		bool compute_plif_matrix(DREAL* penalties_array, INT* Dim, INT numDims);

		inline CPlif** get_PEN()
		{
			return m_PEN;
		}
		inline INT get_num_plifs()
		{
			return m_num_plifs;
		}
		inline INT get_num_limits()
		{
			return m_num_limits;
		}
		inline bool set_num_states(INT num)
		{
			if (!m_num_states || m_num_states==0)
			{
				m_num_states = num; 
				return true;
			}
			else
				return false;
		}
		inline bool set_plif_matrix(CPlifBase** pm)
		{
			if (!m_plif_matrix)
			{
				m_plif_matrix = pm; 
				return true;
			}
			else
				return false;
		}
		inline  CPlifBase** get_plif_matrix()
		{
			return m_plif_matrix;
		}
		inline INT get_num_states()
		{
			return m_num_states;
		}
		inline bool set_dyn_prog(CDynProg* h)
		{
			delete m_dp;
			m_dp = h;
			return true;
		}
		inline CDynProg* get_dyn_prog()
		{
			return m_dp;
		}
	protected:
		CSGInterface* ui;
		CPlif** m_PEN;
		INT m_num_plifs;
		INT m_num_limits;
		INT m_num_states;
		CDynProg* m_dp;
		CPlifBase** m_plif_matrix;
};
#endif //HAVE_SWIG
#endif

