/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifndef __MKLREGRESSION_H__
#define __MKLREGRESSION_H__

#include "lib/common.h"
#include "classifier/svm/MKL.h"

//
//
//		/** update linear component MKL
//		 *
//		 * @param docs docs
//		 * @param label label
//		 * @param active2dnum active2dnum
//		 * @param a a
//		 * @param a_old a old
//		 * @param working2dnum working2dnum
//		 * @param totdoc totdoc
//		 * @param lin lin
//		 * @param aicache ai cache
//		 * @param c c
//		 */
//		virtual void update_linear_component_mkl(
//			int32_t* docs, int32_t *label,
//			int32_t *active2dnum, float64_t *a, float64_t* a_old,
//			int32_t *working2dnum, int32_t totdoc,
//			float64_t *lin, float64_t *aicache, float64_t* c);
//
//		/** update linear component MKL linadd
//		 *
//		 * @param docs docs
//		 * @param label label
//		 * @param active2dnum active2dnum
//		 * @param a a
//		 * @param a_old a old
//		 * @param working2dnum working2dnum
//		 * @param totdoc totdoc
//		 * @param lin lin
//		 * @param aicache ai cache
//		 * @param c c
//		 */
//		virtual void update_linear_component_mkl_linadd(
//			int32_t* docs, int32_t *label,
//			int32_t *active2dnum, float64_t *a, float64_t* a_old,
//			int32_t *working2dnum, int32_t totdoc,
//			float64_t *lin, float64_t *aicache, float64_t* c);
//
//#ifdef USE_CPLEX
//	cleanup_cplex();
//
//	if (get_mkl_enabled())
//		init_cplex();
//#else
//	if (get_mkl_enabled())
//		SG_ERROR( "CPLEX was disabled at compile-time\n");
//#endif
//
#endif //__MKLREGRESSION_H__
