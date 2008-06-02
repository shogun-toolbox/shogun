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


CGUIStructure::CGUIStructure(CSGInterface* ui_) : ui(ui_), m_PEN(NULL), m_num_plifs(0), m_num_limits(0), 
	m_num_states(0), m_dp(NULL), m_plif_matrix(NULL), m_feature_matrix(0,0,0), m_features_dim3(0), 
	m_num_positions(0), m_all_positions(0), m_content_svm_weights(0), m_num_svm_weights(0), 
	m_state_signals(NULL), m_orf_info(NULL), m_use_orf(true), m_mod_words(NULL)
{
}

CGUIStructure::~CGUIStructure()
{
}

bool CGUIStructure::set_plif_struct(INT N, INT M, DREAL* all_limits,
				DREAL* all_penalties, INT* ids, T_STRING<CHAR>* names,
				DREAL* min_values, DREAL* max_values, bool* all_use_cache,
				INT* all_use_svm, T_STRING<CHAR>* all_transform)
{
	// cleanup 
	//SG_PRINT("set_plif_struct, N:%i\n",N);
	for (INT i=0; i<m_num_plifs; i++)	
		delete m_PEN[i];
	delete[] m_PEN;
	m_PEN=NULL;

	// init values
	m_num_plifs=N;
	m_num_limits=M;
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
		if (id>=N)
			SG_ERROR("plif id (%i)  exceeds array length (%i)\n",id,N);
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
			m_num_plifs=0;
			m_num_limits=0;
			return false;
		}
	}

	return true;
}
bool CGUIStructure::compute_plif_matrix(DREAL* penalties_array, INT* Dim, INT numDims)
{
	CPlif** PEN = get_PEN();
	INT num_states = Dim[0];
        if (!set_num_states(Dim[0]))
		return false;
        INT num_plifs = get_num_plifs();

	SG_PRINT("num_states: %i \n",num_states);
	SG_PRINT("dim3: %i \n",Dim[2]);

        CPlifBase **PEN_matrix = new CPlifBase*[num_states*num_states] ;
        CArray3<DREAL> penalties(penalties_array, num_states, num_states, Dim[2], false, false) ;

        for (INT i=0; i<num_states; i++)
        {
                for (INT j=0; j<num_states; j++)
                {
			SG_PRINT(" %.2f ",penalties.get_element(i,j,1));
                        CPlifArray * plif_array = new CPlifArray() ;
                        CPlif * plif = NULL ;
                        plif_array->clear() ;
                        for (INT k=0; k<Dim[2]; k++)
                        {
                                if (penalties.element(i,j,k)==0)
                                        continue ;
                                INT id = (INT) penalties.element(i,j,k)-1 ;
                                if ((id<0 || id>=num_plifs) && (id!=-1))
                                {
                                        SG_ERROR( "id out of range\n") ;
                                        delete_penalty_struct(PEN, num_plifs) ;
                                        return false ;
                                }
                                plif = PEN[id] ;
                                plif_array->add_plif(plif) ;
                        }
                        if (plif_array->get_num_plifs()==0)
                        {
                                delete plif_array ;
                                PEN_matrix[i+j*num_states] = NULL ;
                        }
                        else if (plif_array->get_num_plifs()==1)
                        {
                                delete plif_array ;
                                ASSERT(plif!=NULL) ;
                                PEN_matrix[i+j*num_states] = plif ;
                        }
                        else
                                PEN_matrix[i+j*num_states] = plif_array ;
                }
		SG_PRINT("\n");
        }
	if (!set_plif_matrix(PEN_matrix))
		return false;
	return true;
}
bool  CGUIStructure::set_signal_plifs(INT* state_signals, INT feat_dim3, INT num_states )
{
	INT Nplif = get_num_plifs();
	CPlif** PEN = get_PEN();

        CPlifBase **PEN_state_signal = new CPlifBase*[feat_dim3*num_states] ;
        for (INT i=0; i<num_states*feat_dim3; i++)
        {
                INT id = (INT) state_signals[i]-1 ;
                if ((id<0 || id>=Nplif) && (id!=-1))
                {
                        SG_ERROR( "id out of range\n") ;
                        delete_penalty_struct(PEN, Nplif) ;
                        return false ;
                }
                if (id==-1)
                        PEN_state_signal[i]=NULL ;
                else
                        PEN_state_signal[i]=PEN[id] ;
        }
	set_state_signals(PEN_state_signal);
	return true;
}
#endif
