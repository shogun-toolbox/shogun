#ifndef FUNCTIONS_H
#define FUNCTIONS_H
/* Include all the necessary headers here */
#include <shogun/optimization/optdefines.h>
namespace shogun
{
class func
{
public:
	/* Solves problems of the form : 0.5*x'*H*x + f'*x 
	 * where H : Hessian , also equal to A'*A
	 * 		 x : Solution vector
	 * 		 f : second coefficient vector
	 * 		 A : collection of all subgradients column-wise
	 * 		 subgrad : contains the present subgradient, i.e. at W corresponding to current x
	 * 		 diag_H : The diagonal matrix of the current Hessian
	 * 		 dim : Dimension of the solution vector.
	 * */
	float64_t* H;
	float64_t* f;
	float64_t* x;
	uint32_t dim;
	float64_t* A;
	float64_t* subgrad;
	float64_t* diag_H;
	uint32_t BufSize;
	
	func();
	int init(uint32_t dim, uint32_t bsize);
	void cleanup();
};
}
#endif
