/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _PLIFMATRIX_H_
#define _PLIFMATRIX_H_

#include "base/SGObject.h"
#include "structure/Plif.h"
#include "structure/PlifBase.h"
#include "features/StringFeatures.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"

class CPlifMatrix: public CSGObject
{
	public:
		CPlifMatrix();
		~CPlifMatrix();

		inline CPlif** get_PEN() { return m_PEN; }
		inline CPlifBase** get_plif_matrix() { return m_plif_matrix; }
		inline CPlifBase** get_state_signals() { return m_state_signals; }

		inline int32_t get_num_plifs() { return m_num_plifs; }
		inline int32_t get_num_limits() { return m_num_limits; }

		/** create an empty plif matrix of size num_plifs * num_limits
		 *
		 * @param num_plifs number of plifs 
		 * @param num_limits number of plif limits
		 */
		void create_plifs(int32_t num_plifs, int32_t num_limits);

		/** set plif ids
		 *
		 * @param plif_id_matrix plif id matrix
		 * @param m dimension m of matrix
		 * @param n dimension n of matrix
		 */
		void set_plif_ids(int32_t* ids, int32_t num_ids);
		void set_plif_min_values(float64_t* plif_values, int32_t num_values);
		void set_plif_max_values(float64_t* plif_values, int32_t num_values);

		bool set_plif_struct(float64_t* all_limits,
				float64_t* all_penalties, T_STRING<char>* names,
				bool* all_use_cache, int32_t* all_use_svm, T_STRING<char>* all_transform);

		bool compute_plif_matrix(
				float64_t* penalties_array, int32_t* Dim, int32_t numDims);
		bool compute_signal_plifs(
				int32_t* state_signals, int32_t feat_dim3, int32_t num_states);



		/** set best path plif state signal matrix
		 *
		 * @param plif_id_matrix plif id matrix
		 * @param m dimension m of matrix
		 * @param n dimension n of matrix
		 */
		void set_plif_state_signal_matrix(int32_t *plif_id_matrix, int32_t m, int32_t n);


		/** @return object name */
		inline virtual const char* get_name() const { return "PlifMatrix"; }

	protected:
		CPlif** m_PEN;
		int32_t m_num_plifs;
		int32_t m_num_limits;

		CArray<int32_t> m_ids;
		CArray<float64_t> m_min_values;
		CArray<float64_t> m_max_values;

		CPlifBase** m_plif_matrix;
		CPlifBase** m_state_signals;
};
#endif /* _PLIFMATRIX_H_ */
