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

#include <lib/config.h>
#include <base/SGObject.h>
#include <structure/Plif.h>
#include <structure/PlifArray.h>
#include <structure/PlifBase.h>
#include <structure/DynProg.h>
#include <structure/PlifMatrix.h>

namespace shogun
{
class CSGInterface;

/** @brief UI structure */
class CGUIStructure : public CSGObject
{
	public:
		/** constructor */
		CGUIStructure() {};
		/** constructor
		 * @param interface
		 */
		CGUIStructure(CSGInterface* interface);
		/** destructor */
		~CGUIStructure();

		/** set dyn prog
		 * @param h
		 */
		inline bool set_dyn_prog(CDynProg* h)
		{
			SG_UNREF(m_dp);
			m_dp = h;
			return true;
		}

		/** get dyn prog */
		inline CDynProg* get_dyn_prog()
		{
			if (!m_dp)
				SG_ERROR("no DynProg object found, use set_model first\n")
			return m_dp;
		}

		/** get feature matrix
		 * @param copy
		 */
		inline float64_t* get_feature_matrix(bool copy)
		{
			if (copy)
			{
				int32_t len = m_feature_dims[0]*m_feature_dims[1]*m_feature_dims[2];
				float64_t* d_cpy = SG_MALLOC(float64_t, len);
				memcpy(d_cpy, m_feature_matrix,len*sizeof(float64_t));
				return d_cpy;
			}
			else
				return m_feature_matrix;
		}

		/** get feature matrix sparse
		 * @param index
		 */
		inline CSparseFeatures<float64_t>* get_feature_matrix_sparse(int32_t index)
		{
			ASSERT(index>=0 && index<=1)
			if (index==0)
				return m_feature_matrix_sparse1;
			if (index==1)
				return m_feature_matrix_sparse2;
			return NULL ;
		}

		/** set feature matrix
		 * @param feat
		 * @param dims
		 */
		inline bool set_feature_matrix(float64_t* feat, int32_t* dims)
		{
			SG_FREE(m_feature_matrix);
			int32_t len = dims[0]*dims[1]*dims[2];
			m_feature_matrix = SG_MALLOC(float64_t, len);
			memcpy(m_feature_matrix, feat, len*sizeof(float64_t));
			return true;
		}

		/** set feature matrix sparse
		 * @param f1
		 * @param f2
		 * @param dims
		 */
		inline bool set_feature_matrix_sparse(SGSparseVector<float64_t> *f1, SGSparseVector<float64_t> *f2, int32_t* dims)
		{
			SG_FREE(m_feature_matrix_sparse1);
			SG_FREE(m_feature_matrix_sparse2);

			m_feature_matrix_sparse1 = new CSparseFeatures<float64_t>(SGSparseMatrix<float64_t>(f1, dims[0], dims[1], true));
			m_feature_matrix_sparse2 = new CSparseFeatures<float64_t>(SGSparseMatrix<float64_t>(f2, dims[0], dims[1], true));

			return true;
		}

		/** set feature dims
		 * @param dims
		 */
		inline bool set_feature_dims(int32_t* dims)
		{
			SG_FREE(m_feature_dims);
			m_feature_dims = SG_MALLOC(int32_t, 3);
			memcpy(m_feature_dims, dims,3*sizeof(int32_t));
			return true;
		}
		/** get feature dims */
		inline int32_t* get_feature_dims() { return m_feature_dims; }

		/** set all pos
		 * @param pos
		 * @param Npos
		 */
		inline bool set_all_pos(int32_t* pos, int32_t Npos)
		{
			if (m_all_positions!=pos)
				SG_FREE(m_all_positions);
			int32_t* cp_array = SG_MALLOC(int32_t, Npos);
			memcpy(cp_array, pos, Npos*sizeof(int32_t));
			m_num_positions = Npos;
			m_all_positions = cp_array;
			return true;
		}
		/** get all positions */
		inline int32_t* get_all_positions() { return m_all_positions; }
		/** get num positions */
		inline int32_t get_num_positions() { return m_num_positions; }

		/** set content svm weights
		 * @param weights
		 * @param Nweights
		 * @param Mweights
		 */
		inline bool set_content_svm_weights(
			float64_t* weights, int32_t Nweights,
			int32_t Mweights /* ==num_svms */)
		{
			if (m_content_svm_weights!=weights)
				SG_FREE(m_content_svm_weights);
			float64_t* cp_array = SG_MALLOC(float64_t, Nweights*Mweights);
			memcpy(cp_array, weights,Nweights*Mweights*sizeof(float64_t));
			m_content_svm_weights = cp_array;
			m_num_svm_weights = Nweights;
			return true;
		}
		/** get content svm weights */
		inline float64_t* get_content_svm_weights() { return m_content_svm_weights; }
		/** get num svm weights */
		inline int32_t get_num_svm_weights() { return m_num_svm_weights; }

		/** get plif matrix */
		inline CPlifMatrix* get_plif_matrix() { return m_plif_matrix; }

		/** set orf info
		 * @param orf_info
		 * @param Norf_info
		 * @param Morf_info
		 */
		inline bool set_orf_info(
			int32_t* orf_info, int32_t Norf_info, int32_t Morf_info)
		{
			if (m_orf_info!=orf_info)
				SG_FREE(m_orf_info);
			int32_t* cp_array = SG_MALLOC(int32_t, Norf_info*Morf_info);
			memcpy(cp_array, orf_info,Norf_info*Morf_info*sizeof(int32_t));
			m_orf_info = cp_array;
			return true;
		}

		/** get orf info */
		inline int32_t* get_orf_info()
		{
			return m_orf_info;
		}

		/** set use orf
		 * @param use_orf
		 */
		inline bool set_use_orf(bool use_orf)
		{
			m_use_orf = use_orf;
			return true;
		}
		/** get use orf */
		inline bool get_use_orf() { return m_use_orf; }

		/** set mod words
		 * @param mod_words
		 * @param Nmod_words
		 * @param Mmod_words
		 */
		inline bool set_mod_words(
			int32_t* mod_words, int32_t Nmod_words, int32_t Mmod_words)
		{
			if (mod_words!=m_mod_words)
				SG_FREE(m_mod_words);
			int32_t* cp_array = SG_MALLOC(int32_t, Nmod_words*Mmod_words);
			memcpy(cp_array, mod_words, Nmod_words*Mmod_words*sizeof(int32_t));
			m_mod_words = cp_array;
			return true;
		}
		/** get mod words */
		inline int32_t* get_mod_words() { return m_mod_words; }
		/** get num states */
		inline int32_t get_num_states() { return m_num_states; }
		/** set num states
		 * @param num
		 */
		inline bool set_num_states(int32_t num)
		{
			m_num_states = num;
			return true;
		}
		/** cleanup */
		inline bool cleanup()
		{
			delete m_dp;
			//SG_FREE(m_feature_matrix);
			//delete m_feature_matrix_sparse1;
			//delete m_feature_matrix_sparse2;
			//SG_FREE(m_feature_dims);
			//SG_FREE(m_all_positions);
			//SG_FREE(m_content_svm_weights);
			//delete m_orf_info;
			//delete m_mod_words;
			//delete m_plif_matrix;

			return true;
		}

		/** @return object name */
		virtual const char* get_name() const { return "GUIStructure"; }

	protected:
		/** ui */
		CSGInterface* ui;
		/** num plifs */
		int32_t m_num_plifs;
		/** num limits */
		int32_t m_num_limits;
		/** num states */
		int32_t m_num_states;
		/** dp */
		CDynProg* m_dp;
		/** feature matrix */
		float64_t* m_feature_matrix;
		/** feature matrix sparse 1 */
		CSparseFeatures<float64_t>* m_feature_matrix_sparse1;
		/** feature matrix sparse 2 */
		CSparseFeatures<float64_t>* m_feature_matrix_sparse2;
		/** feature dims */
		int32_t* m_feature_dims;
		/** num positions */
		int32_t m_num_positions;
		/** all positions */
		int32_t* m_all_positions;
		/** content svm weights */
		float64_t* m_content_svm_weights;
		/** num svm weights */
		int32_t m_num_svm_weights;
		/** orf info */
		int32_t* m_orf_info;
		/** use orf */
		bool m_use_orf;
		/** mod words */
		int32_t* m_mod_words;
		/** plif matrix */
		CPlifMatrix* m_plif_matrix;
};
}
#endif

