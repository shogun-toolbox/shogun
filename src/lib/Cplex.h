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


protected:
  CPXENVptr     env;
  CPXLPptr      lp;
  bool          lp_initialized;
};
#endif
#endif
