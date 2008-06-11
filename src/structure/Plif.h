/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PLIF_H__
#define __PLIF_H__

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "structure/PlifBase.h"

#include "lib/matlab.h"

enum ETransformType
{
	T_LINEAR,
	T_LOG,
	T_LOG_PLUS1,
	T_LOG_PLUS3,
	T_LINEAR_PLUS3
};

/** class Plif */
class CPlif: public CPlifBase
{
	public:
		/** constructor
		 *
		 * @param len len
		 */
		CPlif(INT len=0);
		~CPlif();

		/** init penalty struct cache */
		void init_penalty_struct_cache();

		/** lookup penalty SVM
		 *
		 * @param p_value value
		 * @param d_values d values
		 * @return the penalty
		 */
		DREAL lookup_penalty_svm(DREAL p_value, DREAL *d_values) const;

		/** lookup penalty DREAL
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return the penalty
		 */
		DREAL lookup_penalty(DREAL p_value, DREAL* svm_values) const;

		/** lookup penalty INT
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 * @return the penalty
		 */
		DREAL lookup_penalty(INT p_value, DREAL* svm_values) const;

		/** lookup
		 *
		 * @param p_value value
		 * @return a penalty
		 */
		inline DREAL lookup(DREAL p_value)
		{
			ASSERT(use_svm == 0);
			return lookup_penalty(p_value, NULL);
		}

		/** penalty clear derivative */
		void penalty_clear_derivative();

		/** penalty add derivative SVM
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		void penalty_add_derivative_svm(DREAL p_value, DREAL* svm_values) ;

		/** penalty add derivative
		 *
		 * @param p_value value
		 * @param svm_values SVM values
		 */
		void penalty_add_derivative(DREAL p_value, DREAL* svm_values) ;

		/** get cum derivative
		 *
		 * @param p_len len
		 * @return cum derivative
		 */
		const DREAL * get_cum_derivative(INT & p_len) const
		{
			p_len = len;
			return cum_derivatives;
		}

		/** set transform type
		 *
		 * @param type_str type (string)
		 * @return if setting was succesful
		 */
		bool set_transform_type(const char *type_str);

		/** get transform type
		 *
		 * @return type_str type (string)
		 */
		const char* get_transform_type()
		{
			if (transform== T_LINEAR)
				return "linear";
			else if (transform== T_LOG)
				return "log";
			else if (transform== T_LOG_PLUS1)
				return "log(+1)";
			else if (transform== T_LOG_PLUS3)
				return "log(+3)";
			else if (transform== T_LINEAR_PLUS3)
				return "(+3)";
			else 
				SG_ERROR("wrong type");
			return "";
		}


		/** set ID
		 *
		 * @param p_id the id to set
		 */
		void set_id(INT p_id)
		{
			id=p_id;
		}

		/** get ID
		 *
		 * @return the ID
		 */
		INT get_id() const
		{
			return id;
		}

		/** get maximum ID
		 *
		 * @return maximum ID
		 */
		INT get_max_id() const
		{
			return get_id();
		}

		/** set use SVM
		 *
		 * @param p_use_svm if SVM shall be used
		 */
		void set_use_svm(INT p_use_svm)
		{
			delete[] cache;
			cache=NULL;
			use_svm=p_use_svm;
		}

		/** get use SVM
		 *
		 * @return if SVM is used
		 */
		INT get_use_svm() const
		{
			return use_svm;
		}

		/** check if plif uses SVM values
		 *
		 * @return if plif uses SVM values
		 */
		virtual bool uses_svm_values() const
		{
			return (get_use_svm()!=0);
		}

		/** set use cache
		 *
		 * @param p_use_cache if cache shall be used
		 */
		void set_use_cache(INT p_use_cache)
		{
			delete[] cache;
			cache=NULL;
			use_cache=p_use_cache;
		}
		
		/** get use cache
		 *
		 * @return if cache is used
		 */
		INT get_use_cache()
		{
			return use_cache;
		}

		/** set plif
		 *
		 * for swig use set_plif_len, set_plif_limits, set_plif_penalty
		 *
		 * @param p_len len
		 * @param p_limits limit
		 * @param p_penalties penalties
		 */
		void set_plif(INT p_len, DREAL *p_limits, DREAL* p_penalties)
		{
			len=p_len;
			delete[] limits;
			delete[] penalties;
			delete[] cum_derivatives;
			delete[] cache;
			cache=NULL;

			limits=new DREAL[len];
			penalties=new DREAL[len];
			cum_derivatives=new DREAL[len];

			for (INT i=0; i<len; i++)
			{
				limits[i]=p_limits[i];
				penalties[i]=p_penalties[i];
			}

			penalty_clear_derivative();
		}

		/** set plif length
		 *
		 * @param p_len len
		 */
		void set_plif_length(INT p_len)
		{
			if (len!=p_len)
			{
				len=p_len;
				delete[] limits;
				delete[] penalties;
				delete[] cum_derivatives;
				SG_DEBUG( "set_plif len=%i\n", p_len);
				limits=new DREAL[len];
				penalties=new DREAL[len];
				cum_derivatives=new DREAL[len];
			}
			delete[] cache;
			cache=NULL;
			for (INT i=0; i<len; i++)
			{
				limits[i]=0.0;
				penalties[i]=0.0;
			}
			penalty_clear_derivative();
		}

		/** set plif limits
		 *
		 * @param p_limits limits
		 * @param p_len len
		 */
		void set_plif_limits(DREAL* p_limits, INT p_len)
		{
			delete[] cache;
			cache=NULL;
			ASSERT(len==p_len);

			for (INT i=0; i<len; i++)
				limits[i]=p_limits[i];

			penalty_clear_derivative();
		}

		/** get Plif limits
		 *
		 * @return limits
		 */
		DREAL* get_plif_limits()
		{
			return limits;
		}

		/** set plif penalty
		 *
		 * @param p_penalties penalties
		 * @param p_len len
		 */
		void set_plif_penalty(DREAL* p_penalties, INT p_len)
		{
			delete[] cache;
			cache=NULL;
			ASSERT(len==p_len);

			for (INT i=0; i<len; i++)
				penalties[i]=p_penalties[i];

			penalty_clear_derivative();
		}
		/** get plif penalty
 		 *	
 		 * @return plif penalty
 		 */ 
		DREAL* get_plif_penalties()
		{
			return penalties;
		}
		/** set maximum value
		 *
		 * @param p_max_value maximum value
		 */
		inline void set_max_value(DREAL p_max_value)
		{
			delete[] cache;
			cache=NULL;
			max_value=p_max_value;
		}

		/** get maximum value
		 *
		 * @return maximum value
		 */
		virtual DREAL get_max_value() const
		{
			return max_value;
		}

		/** set minimum value
		 *
		 * @param p_min_value minimum value
		 */
		inline void set_min_value(DREAL p_min_value)
		{
			delete[] cache;
			cache=NULL;
			min_value=p_min_value;
		}

		/** get minimum value
		 *
		 * @return minimum value
		 */
		virtual DREAL get_min_value() const
		{
			return min_value;
		}

		/** set name
		 *
		 * @param p_name name
		 */
		void set_name(char *p_name);

		/** get name
		 *
		 * @return name
		 */
		inline char * get_name() const
		{
			if (name)
				return name;
			else
			{
				char buf[20];
				sprintf(buf, "plif%i", id);
				//name = strdup(buf);
				return strdup(buf);
			}
		}

		/** get do calc
		 *
		 * @return if calc shall be done
		 */
		bool get_do_calc();

		/** set do calc
		 *
		 * @param b if calc shall be done
		 */
		void set_do_calc(bool b);
		
		/** get SVM_ids and number of SVMs used
		 *
		 */
		void get_used_svms(INT* num_svms, INT* svm_ids);
		
		/** get plif len
		 *
		 * @return plif len
		 */
		inline INT get_plif_len()
		{
			return len;
		}

	protected:
		/** len */
		INT len;
		/** limits */
		DREAL *limits;
		/** penalties */
		DREAL *penalties;
		/** cum derivatives */
		DREAL *cum_derivatives;
		/** maximum value */
		DREAL max_value;
		/** minimum value */
		DREAL min_value;
		/** cache */
		DREAL *cache;
		/** transform type */
		enum ETransformType transform;
		/** id */
		INT id;
		/** name */
		char * name;
		/** if SVM shall be used */
		INT use_svm;
		/** if cache shall be used */
		bool use_cache;
		/** do calc
		 *  if this is true: lookup_penalty behaves normal
		 *  else: lookup_penalty returns the p_value untransformed*/
		bool do_calc;
};

#ifdef HAVE_MATLAB
CPlif** read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT P) ;
#endif

void delete_penalty_struct(CPlif** PEN, INT P) ;

#endif
