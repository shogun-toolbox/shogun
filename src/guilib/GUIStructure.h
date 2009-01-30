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

		bool set_plif_struct(
			int32_t N, int32_t M, float64_t* all_limits,
			float64_t* all_penalties, int32_t* ids, T_STRING<char>* names,
			float64_t* min_values, float64_t* max_values, bool* all_use_cache,
			int32_t* all_use_svm, T_STRING<char>* all_transform);

		bool compute_plif_matrix(
			float64_t* penalties_array, int32_t* Dim, int32_t numDims);

		bool set_signal_plifs(
			int32_t* state_signals, int32_t feat_dim3, int32_t num_states);

		inline CPlif** get_PEN() { return m_PEN; }
		inline int32_t get_num_plifs() { return m_num_plifs; }
		inline int32_t get_num_limits() { return m_num_limits; }

		inline bool set_num_states(int32_t num)
		{
			//if (!m_num_states || m_num_states==0)
			m_num_states = num; 
			//else
			//	return false;
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
		//
		inline  CPlifBase** get_plif_matrix() { return m_plif_matrix; }
		inline int32_t get_num_states() { return m_num_states; }

		inline bool set_dyn_prog(CDynProg* h)
		{
			SG_UNREF(m_dp);
			m_dp = h;
			return true;
		}

		inline CDynProg* get_dyn_prog()
		{
			if (!m_dp)
				SG_ERROR("no DynProg object found, use set_model first\n");
			return m_dp;
		}

		inline float64_t* get_feature_matrix(bool copy)
		{
			if (copy)
			{
				int32_t len = m_feature_dims[0]*m_feature_dims[1]*m_feature_dims[2];
				float64_t* d_cpy = new float64_t[len];
				memcpy(d_cpy, m_feature_matrix,len*sizeof(float64_t));
				return d_cpy;
			}
			else 
				return m_feature_matrix;
		}

		inline bool set_feature_matrix(float64_t* feat, int32_t* dims)
		{
			delete[] m_feature_matrix;
			int32_t len = dims[0]*dims[1]*dims[2];
			m_feature_matrix = new float64_t[len];
			memcpy(m_feature_matrix, feat,len*sizeof(float64_t));
			return true;
		}

		inline bool set_feature_dims(int32_t* dims)
		{
			delete[] m_feature_dims;
			m_feature_dims = new int32_t[3];
			memcpy(m_feature_dims, dims,3*sizeof(int32_t));
			return true;
		}
		inline int32_t* get_feature_dims() { return m_feature_dims; }

		inline bool set_all_pos(int32_t* pos, int32_t Npos)
		{
			if (m_all_positions!=pos)
				delete[] m_all_positions;
			int32_t* cp_array = new int32_t[Npos];
			memcpy(cp_array, pos,Npos*sizeof(int32_t));
			m_num_positions = Npos;
			m_all_positions = cp_array;
			return true;
		}
		inline int32_t* get_all_positions() { return m_all_positions; }
		inline int32_t get_num_positions() { return m_num_positions; }

		inline bool set_content_svm_weights(
			float64_t* weights, int32_t Nweights,
			int32_t Mweights /* ==num_svms */)
		{
			if (m_content_svm_weights!=weights)
				delete[] m_content_svm_weights;
			float64_t* cp_array = new float64_t[Nweights*Mweights];
			memcpy(cp_array, weights,Nweights*Mweights*sizeof(float64_t));
			m_content_svm_weights = cp_array;
			m_num_svm_weights = Nweights;
			return true;
		}
		inline float64_t* get_content_svm_weights() { return m_content_svm_weights; }
		inline int32_t get_num_svm_weights() { return m_num_svm_weights; }

		inline bool set_state_signals(CPlifBase** ss)
		{
			m_state_signals = ss;
			return true;
		}
		inline CPlifBase** get_state_signals() { return m_state_signals; }

		inline bool set_orf_info(
			int32_t* orf_info, int32_t Norf_info, int32_t Morf_info)
		{
			if (m_orf_info!=orf_info)
				delete[] m_orf_info;
			int32_t* cp_array = new int32_t[Norf_info*Morf_info];
			memcpy(cp_array, orf_info,Norf_info*Morf_info*sizeof(int32_t));
			m_orf_info = cp_array;
			return true;
		}

		inline int32_t* get_orf_info()
		{
			return m_orf_info;
		}

		inline bool set_use_orf(bool use_orf)
		{
			m_use_orf = use_orf;
			return true;
		}
		inline bool get_use_orf() { return m_use_orf; }

		inline bool set_mod_words(
			int32_t* mod_words, int32_t Nmod_words, int32_t Mmod_words)
		{
			if (mod_words!=m_mod_words)
				delete[] m_mod_words;
			int32_t* cp_array = new int32_t[Nmod_words*Mmod_words];
			memcpy(cp_array, mod_words, Nmod_words*Mmod_words*sizeof(int32_t));
			m_mod_words = cp_array;
			return true;	
		}
		inline int32_t* get_mod_words() { return m_mod_words; }

	protected:
		CSGInterface* ui;
		CPlif** m_PEN;
		int32_t m_num_plifs;
		int32_t m_num_limits;
		int32_t m_num_states;
		CDynProg* m_dp;
		CPlifBase** m_plif_matrix;
		float64_t* m_feature_matrix;
		int32_t* m_feature_dims;
		int32_t m_num_positions;
		int32_t* m_all_positions;
		float64_t* m_content_svm_weights;
		int32_t m_num_svm_weights;
		CPlifBase** m_state_signals;
		int32_t* m_orf_info;
		bool m_use_orf;
		int32_t* m_mod_words;
};
#endif //HAVE_SWIG
#endif

