#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H
#include <shogun/optimization/optdefines.h>
namespace shogun
{
class constraints
{
public:
	/* 
	 * TolRel : Relative Tolerance
	 * TolAbs : Absolute Tolerance
	 * UB  : The upper bound for inequalities/equalities
	 * S   : array for storing whether an equality or inequality.
	 * I   : conditional array.
	 * dim : Dimension of the solution vector.
	 * cp_models : Number of different cutting plane models
	 * */
	 float64_t TolRel, TolAbs;
	 float64_t *UB;
	 uint32_t* I;
	 uint8_t* S;
	 uint32_t dim;
	 uint32_t cp_models;
	 uint32_t BufSize;
	 constraints();
	 int init(float64_t trel, float64_t tabs, uint32_t dims, uint32_t bsize, uint32_t cp_models = 1);
	 void cleanup();
};
}
#endif
