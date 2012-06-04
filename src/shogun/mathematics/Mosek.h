/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _CMOSEK__H__
#define _CMOSEK__H__

#ifdef USE_MOSEK

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

#include "mosek.h"

namespace shogun
{

/** @brief Class CMosek to encapsulate access to the commercial MOSEK
 * purpose optimizer.
 *
 * This class provides methods to set up optimization problems that are
 * used in shogun, e.g. from PrimalMosekSOSVM.
 */
class CMosek : public CSGObject
{

	public:
		/** default constructor */
		CMosek();

		/** destructor */
		~CMosek();

		/** method used to direct the log stream of MOSEK
		 * functions to SG_PRINT
		 *
		 * @param handle function handler
		 * @param str string to print on screen
		 */
		static void MSKAPI print(void* handle, char str[]);

		/** wrapper for MOSEK's function MSK_putaveclist used
		 * to set the values in the linear constraint matrix A
		 *
		 * @param task an optimization task
		 * @param A new linear constraint matrix
		 *
		 * @return MSK result code
		 */
		static MSKrescodee wrapper_putaveclist(MSKtask_t & task, SGMatrix< float64_t > A, int32_t nnza);

		/** wrapper for MOSEK's function MSK_putqobj used to
		 * set the values in the regularization matrix of the 
		 * quadratic objective term
		 *
		 * @param task an optimization task
		 * @param Q0 new regularization matrix, assumed to be
		 * symmetric
		 *
		 * @return MSK result code
		 */
		static MSKrescodee wrapper_putqobj(MSKtask_t & task, SGMatrix< float64_t > Q0);

};

} /* namespace shogun */

#endif /* USE_MOSEK */
#endif /* _CMOSEK__H__ */
