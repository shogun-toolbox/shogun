#include "lib/common.h"

#ifdef USE_CPLEX
#include "lib/Cplex.h"
#include "lib/io.h"
#include <unistd.h>

CCplex::CCplex() : env(NULL), lp(NULL), lp_initialized(false)
{
}

CCplex::~CCplex()
{
	cleanup_cplex();
}

bool CCplex::init_cplex(E_PROB_TYPE typ)
{
	while (env==NULL)
	{
		CIO::message(M_INFO, "trying to initialize CPLEX\n") ;

		int status = 1;
		env = CPXopenCPLEX (&status);

		if ( env == NULL )
		{
			char  errmsg[1024];
			CIO::message(M_WARN, "Could not open CPLEX environment.\n");
			CPXgeterrorstring (env, status, errmsg);
			CIO::message(M_WARN, "%s", errmsg);
			CIO::message(M_WARN, "retrying in 60 seconds\n");
			sleep(60);
		}
		else
		{
			/* Turn on output to the screen */

			status = CPXsetintparam (env, CPX_PARAM_SCRIND, CPX_ON);
			if (status)
				CIO::message(M_ERROR, "Failure to turn on screen indicator, error %d.\n", status);

			if (typ==QP)
				status = CPXsetintparam (env, CPX_PARAM_QPMETHOD, 2);
			else if (typ == LINEAR)
				status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 2);

			if (status)
				CIO::message(M_ERROR, "Failure to select dual lp optimization, error %d.\n", status);
			else
			{
				status = CPXsetintparam (env, CPX_PARAM_DATACHECK, CPX_ON);
				if (status)
					CIO::message(M_ERROR, "Failure to turn on data checking, error %d.\n", status);
				else
				{
					lp = CPXcreateprob (env, &status, "light");

					if ( lp == NULL )
						CIO::message(M_ERROR, "Failed to create optimization problem.\n");
					else
						CPXchgobjsen (env, lp, CPX_MIN);  /* Problem is minimization */
				}
			}
		}
	}

	return (lp != NULL) && (env != NULL);
}

bool CCplex::cleanup_cplex()
{
	bool result=false;

	if (lp)
	{
		INT status = CPXfreeprob(env, &lp);
		lp = NULL;
		lp_initialized = false;

		if (status)
			CIO::message(M_WARN, "CPXfreeprob failed, error code %d.\n", status);
		else
			result = true;
	}

	if (env)
	{
		INT status = CPXcloseCPLEX (&env);
		env=NULL;
		
		if (status)
		{
			char  errmsg[1024];
			CIO::message(M_WARN, "Could not close CPLEX environment.\n");
			CPXgeterrorstring (env, status, errmsg);
			CIO::message(M_WARN, "%s", errmsg);
		}
		else
			result = true;
	}
	return result;
}

bool CCplex::optimize(REAL& sol)
{

	return false;

//   status = CPXcopylp (env, lp, COLSORIG, ROWSSUB, CPX_MIN, Hcost, Hrhs, 
 //                      Hsense, Hmatbeg, Hmatcnt, Hmatind, Hmatval, 
  //                     Hlb, Hub, NULL);

//	int      solnstat, solnmethod, solntype;
//	double   objval, maxviol;
//
//	int status = CPXqpopt (env, lp);
//	if (status)
//		CIO::message(M_ERROR, "Failed to optimize QP.\n");
//
//	solnstat = CPXgetstat (env, lp);
//
//	if ( solnstat == CPX_STAT_UNBOUNDED )
//		CIO::message(M_INFO, "Model is unbounded\n");
//	else if ( solnstat == CPX_STAT_INFEASIBLE )
//		CIO::message(M_INFO, "Model is infeasible\n");
//	else if ( solnstat == CPX_STAT_INForUNBD )
//		CIO::message(M_INFO, "Model is infeasible or unbounded\n");
//
//	status = CPXsolninfo (env, lp, &solnmethod, &solntype, NULL, NULL);
//	if ( status )
//		CIO::message(M_ERROR, "Failed to obtain solution info.\n");
//
//	CIO::message(M_INFO, "Solution status %d, solution method %d\n", solnstat, solnmethod);
//
//	if ( solntype == CPX_NO_SOLN )
//		CIO::message(M_ERROR, "Solution not available.\n");
//
//	status = CPXgetobjval (env, lp, &objval);
//	if ( status )
//		CIO::message(M_ERROR, "Failed to obtain objective value.\n");
//	CIO::message(M_INFO, "Objective value %.10g.\n", objval);
//
	///* The size of the problem should be obtained by asking CPLEX what
	//   the actual size is.  cur_numrows and cur_numcols store the 
	//   current number of rows and columns, respectively.  */

	//cur_numcols = CPXgetnumcols (env, lp);
	//cur_numrows = CPXgetnumrows (env, lp);

	///* Retrieve basis, if one is available */

	//if ( solntype == CPX_BASIC_SOLN ) {
	//	cstat = (int *) malloc (cur_numcols*sizeof(int));
	//	rstat = (int *) malloc (cur_numrows*sizeof(int));
	//	if ( cstat == NULL || rstat == NULL ) {
	//		fprintf (stderr, "No memory for basis statuses.\n");
	//		goto TERMINATE;
	//	}

	//	status = CPXgetbase (env, lp, cstat, rstat);
	//	if ( status ) {
	//		fprintf (stderr, "Failed to get basis; error %d.\n", status);
	//		goto TERMINATE;
	//	}
	//}
	//return (status==0);
}
#endif
