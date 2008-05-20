/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "lib/io.h"

#include "interface/SGInterface.h"
#include "guilib/GUIStructure.h"


CGUIStructure::CGUIStructure(CSGInterface* ui_) : ui(ui_), m_PEN(NULL), m_N(0), m_M(0)
{
}

CGUIStructure::~CGUIStructure()
{
}

bool CGUIStructure::set_plif_struct(INT N, INT M, DREAL* all_limits,
				DREAL* all_penalties, INT* ids, T_STRING<CHAR>* names,
				DREAL* min_values, DREAL* max_values, bool* all_use_cache,
				bool* all_use_svm, T_STRING<CHAR>* all_transform)
{
	// cleanup 
	for (INT i=0; i<m_N; i++)	
		delete[] m_PEN[i];
	delete m_PEN;
	m_PEN=NULL;
	m_N=0;

	// init values
	m_N=N;
	m_M=M;
	m_PEN = new CPlif*[N] ;
	for (INT i=0; i<N; i++)	
		m_PEN[i]=new CPlif() ;

	for (INT i=0; i<N; i++)
	{
		DREAL* limits = new DREAL[M];
		DREAL* penalties = new DREAL[M];
		for (INT k=0; k<M; k++)
		{
			limits[k] = all_limits[i*M+k];
			penalties[k] = all_penalties[i*M+k];
		}
		INT id = ids[i];

		m_PEN[id]->set_id(id);

		m_PEN[id]->set_name(get_zero_terminated_string_copy(names[i]));
		m_PEN[id]->set_min_value(min_values[i]);
		m_PEN[id]->set_max_value(max_values[i]);
		m_PEN[id]->set_use_cache(all_use_cache[i]);
		m_PEN[id]->set_use_svm(all_use_svm[i]);
		m_PEN[id]->set_plif(M,limits,penalties);
		//m_PEN[id]->set_do_calc(all_do_calc[i]); //JONAS FIX
		CHAR* transform_str=get_zero_terminated_string_copy(all_transform[i]);
		if (!m_PEN[id]->set_transform_type(transform_str))
		{
			SG_ERROR( "transform type not recognized ('%s')\n", transform_str) ;
			delete[] m_PEN;
			m_PEN=NULL;
			m_N=0;
			m_M=0;
			return false;
		}
	}

	return true;
}
#endif
