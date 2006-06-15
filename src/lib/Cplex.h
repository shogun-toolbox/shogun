/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef CCPLEX_H__
#define CCPLEX_H__

#include "lib/common.h"

#ifdef USE_CPLEX
extern "C" {
#include <ilcplex/cplex.h>
}

class CCplex
{
public:
	enum E_PROB_TYPE
	{
		LINEAR,
		QP
	};

	CCplex();
	~CCplex();

	bool init_cplex(E_PROB_TYPE t);
	bool cleanup_cplex();
	bool optimize(DREAL& sol);


protected:
  CPXENVptr     env;
  CPXLPptr      lp;
  bool          lp_initialized;
};
#endif
#endif
