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

class CPlifMatrix: public CSGObject
{
	public:
		CPlifMatrix();
		~CPlifMatrix();

		inline CPlif** get_PEN() { return m_PEN; }
		inline  CPlifBase** get_plif_matrix() { return m_plif_matrix; }

		inline int32_t get_num_plifs() { return m_num_plifs; }
		inline int32_t get_num_limits() { return m_num_limits; }


		bool set_plif_struct(
				int32_t N, int32_t M, float64_t* all_limits,
				float64_t* all_penalties, int32_t* ids, T_STRING<char>* names,
				float64_t* min_values, float64_t* max_values, bool* all_use_cache,
				int32_t* all_use_svm, T_STRING<char>* all_transform);

		bool compute_plif_matrix(
				float64_t* penalties_array, int32_t* Dim, int32_t numDims);
		bool compute_signal_plifs(
			int32_t* state_signals, int32_t feat_dim3, int32_t num_states);
		inline bool set_state_signals(CPlifBase** ss)
		{
			m_state_signals = ss;
			return true;
		}

		inline CPlifBase** get_state_signals() { return m_state_signals; }

		/** @return object name */
		inline virtual const char* get_name() const { return "PlifMatrix"; }

	protected:
		CPlif** m_PEN;
		int32_t m_num_plifs;
		int32_t m_num_limits;

		CPlifBase** m_plif_matrix;
		CPlifBase** m_state_signals;
};
#endif /* _PLIFMATRIX_H_ */
