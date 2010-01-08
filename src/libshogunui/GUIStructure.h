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

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/structure/Plif.h>
#include <shogun/structure/PlifArray.h>
#include <shogun/structure/PlifBase.h>
#include <shogun/structure/DynProg.h>
#include <shogun/structure/PlifMatrix.h>

namespace shogun
{
class CSGInterface;

class CGUIStructure : public CSGObject
{
	public:
		CGUIStructure(CSGInterface* interface);
		~CGUIStructure();

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

		inline CSparseFeatures<float64_t>* get_feature_matrix_sparse(int32_t index)
		{
			ASSERT(index>=0 && index<=1) ;
			if (index==0)
				return m_feature_matrix_sparse1;
			if (index==1)
				return m_feature_matrix_sparse2;
			return NULL ;
		}

		inline bool set_feature_matrix(float64_t* feat, int32_t* dims)
		{
			delete[] m_feature_matrix;
			int32_t len = dims[0]*dims[1]*dims[2];
			m_feature_matrix = new float64_t[len];
			memcpy(m_feature_matrix, feat, len*sizeof(float64_t));
			return true;
		}

		inline bool set_feature_matrix_sparse(TSparse<float64_t> *f1, TSparse<float64_t> *f2, int32_t* dims)
		{
			delete[] m_feature_matrix_sparse1 ;
			delete[] m_feature_matrix_sparse2 ;

			m_feature_matrix_sparse1 = new CSparseFeatures<float64_t>(f1, dims[0], dims[1], true) ;
			m_feature_matrix_sparse2 = new CSparseFeatures<float64_t>(f2, dims[0], dims[1], true) ;

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
			memcpy(cp_array, pos, Npos*sizeof(int32_t));
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

		inline CPlifMatrix* get_plif_matrix() { return m_plif_matrix; }

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
		inline int32_t get_num_states() { return m_num_states; }
		inline bool set_num_states(int32_t num)
		{
			m_num_states = num; 
			return true;
		}

		inline bool cleanup()
		{
			delete m_dp;
			//delete[] m_feature_matrix;
			//delete m_feature_matrix_sparse1;
			//delete m_feature_matrix_sparse2;
			//delete[] m_feature_dims;
			//delete[] m_all_positions;
			//delete[] m_content_svm_weights;
			//delete m_orf_info;
			//delete m_mod_words;
			//delete m_plif_matrix;

			return true;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "GUIStructure"; }

	protected:
		CSGInterface* ui;
		int32_t m_num_plifs;
		int32_t m_num_limits;
		int32_t m_num_states;
		CDynProg* m_dp;
		float64_t* m_feature_matrix;
		CSparseFeatures<float64_t>* m_feature_matrix_sparse1;
		CSparseFeatures<float64_t>* m_feature_matrix_sparse2;
		int32_t* m_feature_dims;
		int32_t m_num_positions;
		int32_t* m_all_positions;
		float64_t* m_content_svm_weights;
		int32_t m_num_svm_weights;
		int32_t* m_orf_info;
		bool m_use_orf;
		int32_t* m_mod_words;
		CPlifMatrix* m_plif_matrix;
};
}
#endif

