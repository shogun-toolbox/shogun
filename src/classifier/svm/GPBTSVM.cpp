#include "classifier/svm/GPBTSVM.h"
#include "classifier/svm/gpdt.h"
#include "lib/io.h"

CGPBTSVM::CGPBTSVM()
{
	model=NULL;
}

CGPBTSVM::~CGPBTSVM()
{
	free(model);
}

bool CGPBTSVM::train()
{
	double     *solution;                     /* store the solution found       */
	QPproblem  prob;                          /* object containing the solvers  */

	assert(get_kernel());
	assert(get_labels() && get_labels()->get_num_labels());
	prob.KER=new sKernel(get_kernel(), get_labels()->get_num_labels());
	assert(prob.KER);
	prob.ell=get_labels()->get_num_labels();
	CIO::message(M_INFO, "%d trainlabels\n", prob.ell);

	//  /*** set options defaults ***/
	//  sigma                = 1.0;
	//  degree               = 3.0;
	//  normalisation        = 1.0;
	//  c_poly               = 1.0;
	//  nOutputStream        = 0;
	prob.verbosity       = 1;
	prob.preprocess_size = -1;

	if (prob.chunk_size < 2)      prob.chunk_size = 2;
	if (prob.q <= 0)              prob.q = prob.chunk_size / 3;
	if (prob.q < 2)               prob.q = 2;
	if (prob.q > prob.chunk_size) prob.q = prob.chunk_size;
	prob.q = prob.q & (~1);
	if (prob.maxmw < 5)
		prob.maxmw = 5;

	/*** set the problem description for final report ***/
	CIO::message(M_INFO, "\nTRAINING PARAMETERS:\n");
	CIO::message(M_INFO, "\tNumber of training documents: %d\n", prob.ell);
	CIO::message(M_INFO, "\tq: %d\n", prob.chunk_size);
	CIO::message(M_INFO, "\tn: %d\n", prob.q);
	CIO::message(M_INFO, "\tC: %lf\n", prob.c_const);
	CIO::message(M_INFO, "\tkernel type: %d\n", prob.ker_type);
	CIO::message(M_INFO, "\tcache size: %dMb\n", prob.maxmw);
	CIO::message(M_INFO, "\tStopping tolerance: %lf\n", prob.delta);

	//  /*** compute the number of cache rows up to maxmw Mb. ***/
	if (prob.preprocess_size == -1)
		prob.preprocess_size = (int) ( (double)prob.chunk_size * 1.5 );

	/*** compute the problem solution *******************************************/
	solution = (double *)malloc(prob.ell * sizeof(double));
	prob.gpdtsolve(solution);
	/****************************************************************************/


	int num_sv=0;
	int bsv;
	int i=0;
	int k=0;

	for (i = 0; i < prob.ell; i++)
	{
		if (solution[i] > prob.DELTAsv)
		{
			num_sv++;
			if (solution[i] > (prob.c_const - prob.DELTAsv)) bsv++;
		}
	}

	create_new_model(num_sv);
	set_bias(-prob.bee);

	CIO::message(M_INFO,"SV: %d BSV = %d\n", num_sv, bsv);

	for (i = 0; i < prob.ell; i++)
	{
		if (solution[i] > prob.DELTAsv)
		{
			set_support_vector(k++, i);
			set_alpha(i, solution[i]*get_labels()->get_label(i));
		}
	}

	free(solution);

	return true;
}
