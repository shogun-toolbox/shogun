/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_BASE_H__
#define __PLIF_BASE_H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/Mathematics.h"

/** class PlifBase */
class CPlifBase : public CSGObject
{
	public:
		/** default constructor */
		CPlifBase() {};
		virtual ~CPlifBase() {};

		/** lookup penalty DREAL
		 *
		 * abstract base method
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return penalty
		 */
		virtual DREAL lookup_penalty(DREAL p_value, DREAL* svm_values) const =0;

		/** lookup penalty INT
		 *
		 * abstract base method
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return penalty
		 */
		virtual DREAL lookup_penalty(INT p_value, DREAL* svm_values) const =0;

		/** penalty clear derivative
		 *
		 * abstrace base method
		 */
		virtual void penalty_clear_derivative()=0;

		/** penalty add derivative
		 *
		 * abstract base method
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		virtual void penalty_add_derivative(DREAL p_value, DREAL* svm_values)=0 ;

		/** get maximum value
		 *
		 * abstract base method
		 *
		 * @return maximum value
		 */
		virtual DREAL get_max_value() const = 0;

		/** get minimum value
		 *
		 * abstract base method
		 *
		 * @return minimum value
		 */
		virtual DREAL get_min_value() const = 0;

		/** get SVM_ids and number of SVMs used
		 *
		 * abstract base method
		 */
		virtual void get_used_svms(INT* num_svms, INT* svm_ids) = 0;

		/** if plif uses SVM values
		 *
		 * abstract base method
		 *
		 * @return if plif uses SVM values
		 */
		virtual bool uses_svm_values() const = 0;

		/** get maximum ID
		 *
		 * abstract base method
		 *
		 * @return maximum ID
		 */
		virtual INT get_max_id() const = 0;
};
#endif
