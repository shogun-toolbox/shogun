/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIFARRAY_H__
#define __PLIFARRAY_H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/DynamicArray.h"
#include "structure/PlifBase.h"

/** class PlifArray */
class CPlifArray: public CPlifBase
{
	public:
		/** default constructor */
		CPlifArray();
		virtual ~CPlifArray();

		/** add plif
		 *
		 * @param new_plif the new plif to be added
		 */
		void add_plif(CPlifBase* new_plif);

		/** clear */
		void clear();

		/** get number of plifs
		 *
		 * @return number of plifs
		 */
		INT get_num_plifs()
		{
			return m_array.get_num_elements();
		}

		/** lookup penalty DREAL
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		virtual DREAL lookup_penalty(DREAL p_value, DREAL* svm_values) const;

		/** lookup penalty INT
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		virtual DREAL lookup_penalty(INT p_value, DREAL* svm_values) const;

		/** penalty clear derivative */
		virtual void penalty_clear_derivative();

		/** penalty add derivative
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		virtual void penalty_add_derivative(DREAL p_value, DREAL* svm_values);

		/** get maximum value
		 *
		 * @return maximum value
		 */
		virtual DREAL get_max_value() const
		{
			return max_value;
		}

		/** get minimum value
		 *
		 * @return minumum value
		 */
		virtual DREAL get_min_value() const
		{
			return min_value;
		}

		/** check if plif uses SVM values
		 *
		 * @return if plif uses SVM values
		 */
		virtual bool uses_svm_values() const;

		/** get maximum ID
		 *
		 * @return maximum ID
		 */
		virtual INT get_max_id() const;

	protected:
		/** plif array */
		CDynamicArray<CPlifBase*> m_array;
		/** maximum value */
		DREAL max_value;
		/** minimum value */
		DREAL min_value;
};

#endif
