/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __PENALTY_INFO_H__
#define __PENALTY_INFO_H__

#include "lib/common.h"
#include "lib/Mathmatics.h"

#ifdef HAVE_MATLAB
#include <mex.h>
#endif

enum ETransformType
{
	T_LINEAR,
	T_LOG,
	T_LOG_PLUS1,
	T_LOG_PLUS3,
	T_LINEAR_PLUS3
}  ;

struct penalty_struct
{
	INT len ;
	DREAL *limits ;
	DREAL *penalties ;
	INT max_len ;
	INT min_len ;
	DREAL *cache ;
	enum ETransformType transform ;
	INT id ;
	struct penalty_struct *next_pen ;
	char * name ;
	INT use_svm ;
} ;

void init_penalty_struct(struct penalty_struct &PEN) ;
void delete_penalty_struct(struct penalty_struct &PEN) ;
void delete_penalty_struct_array(struct penalty_struct *PEN, INT len) ;

#ifdef HAVE_MATLAB
struct penalty_struct * read_penalty_struct_from_cell(const mxArray * mx_penalty_info, INT &P) ;
#endif

DREAL lookup_penalty(const struct penalty_struct *PEN, INT p_value, 
					DREAL* svm_values, bool follow_next, DREAL &input_value) ;

#endif
