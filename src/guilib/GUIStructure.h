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

		bool set_signal_plifs(INT* state_signals, INT feat_dim3, INT num_states );

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
				m_num_states = num; 
			else
				return false;
			return true;
		}
		//inline bool set_plif_matrix(CPlifBase** pm)
		//{
		//	if (!m_plif_matrix)
		//		m_plif_matrix = pm; 
		//	else
		//		return false;
		//	return true;
		//}
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
		inline CArray3<DREAL> get_feature_matrix()
		{
			INT d1,d2,d3;
			m_feature_matrix.get_array_size(d1,d2,d3);
			SG_PRINT("ui: get_features: d1:%i d2:%i d3:%i\n",d1,d2,d3);
			return m_feature_matrix;
		}
		inline bool set_feature_matrix(CArray3<DREAL> feat)
		{
			INT d1,d2,d3;
			feat.get_array_size(d1,d2,d3);
			SG_PRINT("ui: set_features: d1:%i d2:%i d3:%i\n",d1,d2,d3);
			DREAL* cp_array = new DREAL[d1*d2*d3];
			memcpy(cp_array, feat.get_array(),d1*d2*d3*sizeof(DREAL));
			bool copy=false;
			m_feature_matrix.set_array(cp_array, d1, d2, d3, false, copy);
			return true;
		}
		inline bool set_features_dim3(INT num)
		{
			m_features_dim3 = num;
			return true;
		}
		inline INT get_features_dim3()
		{
			return m_features_dim3;
		}
		inline bool set_all_pos(INT* pos, INT Npos)
		{
			INT* cp_array = new INT[Npos];
			memcpy(cp_array, pos,Npos*sizeof(INT));
			m_num_positions = Npos;
			m_all_positions = cp_array;
			return true;	
		}
		inline INT* get_all_positions()
		{
			return m_all_positions;
		}
		inline INT get_num_positions()
		{
			return m_num_positions;
		}
		inline bool set_content_svm_weights(DREAL* weights, INT Nweights, INT Mweights/*==num_svms*/)
		{
			DREAL* cp_array = new DREAL[Nweights*Mweights];
			memcpy(cp_array, weights,Nweights*Mweights*sizeof(DREAL));
			if (!m_content_svm_weights || *m_content_svm_weights==0)
			{
				m_content_svm_weights = cp_array;
				m_num_svm_weights = Nweights;
				return true;	
			}
			return false;
		}
		inline DREAL* get_content_svm_weights()
		{
			return m_content_svm_weights;
		}
		inline INT get_num_svm_weights()
		{
			return m_num_svm_weights;
		}
		inline bool set_state_signals(CPlifBase** ss)
		{
			m_state_signals = ss;
			return true;
		}
		inline CPlifBase** get_state_signals()
		{
			return m_state_signals;
		}
		inline bool set_orf_info(INT* orf_info, INT Norf_info, INT Morf_info)
		{
			INT* cp_array = new INT[Norf_info*Morf_info];
			memcpy(cp_array, orf_info,Norf_info*Morf_info*sizeof(INT));
			m_orf_info = cp_array;
			return true;	
		}
		inline INT* get_orf_info()
		{
			return m_orf_info;
		}

		inline bool set_use_orf(bool use_orf)
		{
			m_use_orf = use_orf;
			return true;	
		}
		inline bool get_use_orf()
		{
			return m_use_orf;
		}
		inline bool set_mod_words(INT* mod_words, INT Nmod_words, INT Mmod_words)
		{
			INT* cp_array = new INT[Nmod_words*Mmod_words];
			memcpy(cp_array, mod_words, Nmod_words*Mmod_words*sizeof(INT));
			m_mod_words = cp_array;
			return true;	
		}
		inline INT* get_mod_words()
		{
			return m_mod_words;
		}

	protected:
		CSGInterface* ui;
		CPlif** m_PEN;
		INT m_num_plifs;
		INT m_num_limits;
		INT m_num_states;
		CDynProg* m_dp;
		CPlifBase** m_plif_matrix;
		CArray3<DREAL> m_feature_matrix;
		INT m_features_dim3;
		INT m_num_positions;
		INT* m_all_positions;
		DREAL* m_content_svm_weights;
		INT m_num_svm_weights;
		CPlifBase** m_state_signals;
		INT* m_orf_info;
		bool m_use_orf;
		INT* m_mod_words;
};
#endif //HAVE_SWIG
#endif

