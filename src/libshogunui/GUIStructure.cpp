/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "SGInterface.h"
#include "GUIStructure.h"

#include <shogun/lib/config.h>
#include <shogun/lib/io.h>
#include <shogun/structure/Plif.h>

CGUIStructure::CGUIStructure(CSGInterface* ui_)
: ui(ui_), m_PEN(NULL), m_num_plifs(0), m_num_limits(0),
	m_num_states(0), m_dp(NULL), m_plif_matrix(NULL), m_feature_matrix(NULL),
	m_feature_dims(NULL), m_num_positions(0), m_all_positions(0),
	m_content_svm_weights(0), m_num_svm_weights(0), m_state_signals(NULL),
	m_orf_info(NULL), m_use_orf(true), m_mod_words(NULL)
{
}

CGUIStructure::~CGUIStructure()
{
}

bool CGUIStructure::set_plif_struct(
	int32_t N, int32_t M, float64_t* all_limits, float64_t* all_penalties,
	int32_t* ids, T_STRING<char>* names, float64_t* min_values,
	float64_t* max_values, bool* all_use_cache, int32_t* all_use_svm,
	T_STRING<char>* all_transform)
{
	// cleanup 
	//SG_PRINT("set_plif_struct, N:%i\n",N);
	for (int32_t i=0; i<m_num_plifs; i++)	
		delete m_PEN[i];
	delete[] m_PEN;
	m_PEN=NULL;

	// init values
	m_num_plifs=N;
	m_num_limits=M;
	m_PEN = new CPlif*[N] ;
	for (int32_t i=0; i<N; i++)	
		m_PEN[i]=new CPlif() ;

	for (int32_t i=0; i<N; i++)
	{
		float64_t* limits = new float64_t[M];
		float64_t* penalties = new float64_t[M];
		for (int32_t k=0; k<M; k++)
		{
			limits[k] = all_limits[i*M+k];
			penalties[k] = all_penalties[i*M+k];
		}
		int32_t id = ids[i];
		if (id>=N)
			SG_ERROR("plif id (%i)  exceeds array length (%i)\n",id,N);
		m_PEN[id]->set_id(id);

		m_PEN[id]->set_plif_name(get_zero_terminated_string_copy(names[i]));
		m_PEN[id]->set_min_value(min_values[i]);
		m_PEN[id]->set_max_value(max_values[i]);
		m_PEN[id]->set_use_cache(all_use_cache[i]);
		m_PEN[id]->set_use_svm(all_use_svm[i]);
		m_PEN[id]->set_plif_limits(limits, M);
		m_PEN[id]->set_plif_penalty(penalties, M);
		//m_PEN[id]->set_do_calc(all_do_calc[i]); //JONAS FIX
		char* transform_str=get_zero_terminated_string_copy(all_transform[i]);
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

bool CGUIStructure::compute_plif_matrix(
	float64_t* penalties_array, int32_t* Dim, int32_t numDims)
{
	CPlif** PEN = get_PEN();
	int32_t num_states = Dim[0];
        if (!set_num_states(Dim[0]))
		return false;
        int32_t num_plifs = get_num_plifs();

	//SG_PRINT("num_states: %i \n",num_states);
	//SG_PRINT("dim3: %i \n",Dim[2]);

	delete[] m_plif_matrix ;
        m_plif_matrix = new CPlifBase*[num_states*num_states] ;
	//SG_PRINT("m_plif_matrix: %p \n",m_plif_matrix);
        CArray3<float64_t> penalties(penalties_array, num_states, num_states, Dim[2], false, true) ;

        for (int32_t i=0; i<num_states; i++)
        {
                for (int32_t j=0; j<num_states; j++)
                {
			//SG_PRINT(" %.2f ",penalties.get_element(i,j,1));
                        CPlifArray * plif_array = new CPlifArray() ;
                        CPlif * plif = NULL ;
                        plif_array->clear() ;
                        for (int32_t k=0; k<Dim[2]; k++)
                        {
                                if (penalties.element(i,j,k)==0)
                                        continue ;
                                int32_t id = (int32_t) penalties.element(i,j,k)-1 ;
				//SG_PRINT("i:%i j:%i k:%i id:%i \n",i, j, k,id);
                                if ((id<0 || id>=num_plifs) && (id!=-1))
                                {
                                        SG_ERROR( "id out of range\n") ;
                                        delete_penalty_struct(PEN, num_plifs) ;
                                        return false ;
                                }
                                plif = PEN[id] ;
				//SG_PRINT("PEN[%i]->get_min_value(): %f\n",id,PEN[id]->get_min_value());
                                plif_array->add_plif(plif) ;
                        }
                        if (plif_array->get_num_plifs()==0)
                        {
                                SG_UNREF(plif_array);
                                m_plif_matrix[i+j*num_states] = NULL ;
                        }
                        else if (plif_array->get_num_plifs()==1)
                        {
                                SG_UNREF(plif_array);
                                ASSERT(plif!=NULL) ;
                                m_plif_matrix[i+j*num_states] = plif ;
                        }
                        else
			{
                                m_plif_matrix[i+j*num_states] = plif_array ;
				//int32_t num_svms;
				//int32_t* used_svms;
				//m_plif_matrix[i+j*num_states]->get_used_svms(&num_svms,used_svms);
			}

                }
		//SG_PRINT("\n");
        }
//	float64_t tmp[] = {0,0,0,0,0,0,0,0,0};
//	for (int32_t i=0;i<num_states;i++)
//                for (int32_t j=0; j<num_states; j++)
//			if (m_plif_matrix[i+j*num_states]!=NULL)
//				SG_PRINT("1 m_plif_matrix[%i]->lookup_penalty(): %f\n",i+j*num_states, m_plif_matrix[i+j*num_states]->lookup_penalty(0,tmp));
	return true;
}

bool  CGUIStructure::set_signal_plifs(
	int32_t* state_signals, int32_t feat_dim3, int32_t num_states)
{
	int32_t Nplif = get_num_plifs();
	CPlif** PEN = get_PEN();

        CPlifBase **PEN_state_signal = new CPlifBase*[feat_dim3*num_states] ;
        for (int32_t i=0; i<num_states*feat_dim3; i++)
        {
                int32_t id = (int32_t) state_signals[i]-1 ;
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
