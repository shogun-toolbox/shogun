#include "lib/common.h"

#ifdef USE_CPLEX
#include "lib/Cplex.h"
#include "lib/io.h"
#include <unistd.h>

CCplex::CCplex() : env(NULL), lp(NULL), lp_initialized(false)
{
}

bool CCplex::init_cplex(E_PROB_TYPE typ)
{
	while (env==NULL)
	{
		CIO::message(M_INFO, "trying to initialize CPLEX\n") ;

		int status = 0;
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
			status = CPXsetintparam (env, CPX_PARAM_LPMETHOD, 2);
			if ( status )
			{
				CIO::message(M_ERROR, 
						"Failure to select dual lp optimization, error %d.\n", status);
			}
			else
			{
				status = CPXsetintparam (env, CPX_PARAM_DATACHECK, CPX_ON);
				if ( status )
				{
					CIO::message(M_ERROR,
							"Failure to turn on data checking, error %d.\n", status);
				}	
				else
				{
					lp = CPXcreateprob (env, &status, "light");

					if ( lp == NULL )
						CIO::message(M_ERROR, "Failed to create LP.\n");
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
#endif
