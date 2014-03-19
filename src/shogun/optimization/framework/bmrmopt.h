#ifndef BMRMOPT_H
#define BMRMOPT_H

#include <shogun/optimization/optdefines.h>
#include <shogun/optimization/functions.h>
#include <shogun/optimization/constraints.h>
#include <shogun/optimization/CP.h>

extern float64_t* Hs;
extern uint32_t BSize;

namespace shogun
{
class bmrmOptimizer
{
	func fx;
	constraints cx;
	CP cps;
	BmrmStatistics bmrm;
	float64_t *W, *prevW, lambda;
	uint32_t BufSize, histSize;
	bool verbose, cleanICP;
	uint32_t cleanAfter;
	CDualLibQPBMSOSVM  *mach;
	
	bmrmOptimizer();
	//Contain both the functions and the constraints.
	int setup(
		CDualLibQPBMSOSVM  *machine,
		float64_t*       Wt,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cICP,
		uint32_t         cAfter,
		bool             vrbose);
	void computeAndUpdate(bool tune_alpha);
	void updateHessian();
	libqp_state_T solveQP(bool tune_alpha);
	void updateWeights();
	void computeRandS(libqp_state_T qps);
	void removeICPs();
	int returnOptima(
	CDualLibQPBMSOSVM  *machine,
		float64_t*       Wt,
		float64_t        TolRel,
		float64_t        TolAbs,
		float64_t        _lambda,
		uint32_t         _BufSize,
		bool             cICP,
		uint32_t         cAfter,
		bool             vrbose
	);
	void cleanup();
};
}

#endif
