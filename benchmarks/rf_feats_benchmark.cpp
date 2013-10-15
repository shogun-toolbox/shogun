#include <shogun/base/init.h>
#include <shogun/features/RandomFourierDotFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/normalizer/IdentityKernelNormalizer.h>
#include <shogun/lib/Time.h>

using namespace shogun;

/** Benchmark code for the RandomFourierDotFeatures class
 * Current results are after the code
 */

int main(int argv, char** argc)
{
	init_shogun_with_defaults();

	int32_t dims[] = {100, 300, 600};
	CTime* timer = new CTime();
	for (index_t d=0; d<3; d++)
	{
		int32_t num_dim = dims[d];
		int32_t num_vecs = 100000;
		SG_SPRINT("-------------------------------------------------------------------------\n");
		SG_SPRINT("Starting experiment for number of dimensions = %d, number of vectors = %d,", num_dim, num_vecs);
		SGMatrix<float64_t> mat(num_dim, num_vecs);
		for (index_t i=0; i<num_vecs; i++)
		{
			for (index_t j=0; j<num_dim; j++)
			{
				mat(j,i) = CMath::random(0,1) + 0.5;
			}
		}

		SGVector<float64_t> params(1);
		params[0] = num_dim - 20;
		SG_SPRINT(" using kernel_width = %f\n", params[0]);

		CDenseFeatures<float64_t>* dense_feats = new CDenseFeatures<float64_t>(mat);
		SG_REF(dense_feats);

		int D[] = {50, 100, 200, 300, 400, 500};
		for (index_t i=0; i<6; i++)
		{
			SG_SPRINT("Results for D = %d\n", D[i]);
			CRandomFourierDotFeatures* rand_feats =
					new CRandomFourierDotFeatures(dense_feats, D[i], KernelName::GAUSSIAN, params);
			rand_feats->benchmark_dense_dot_range();
			rand_feats->benchmark_add_to_dense_vector();
			SG_UNREF(rand_feats);
		}

		SG_SPRINT("-------------------------------------------------------------------------\n");
		SG_UNREF(dense_feats);
	}
	SG_SPRINT("Total time : %fs\n", timer->cur_runtime_diff_sec());
	timer->stop();
	SG_UNREF(timer);

	exit_shogun();
}

/** Current results, using Release settings, for future comparisons :
 * -------------------------------------------------------------------------
 *  Starting experiment for number of dimensions = 100, number of vectors = 100000, using kernel_width = 80.000000
 *  Results for D = 50
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 1.846000s walltime 0.310587s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 1.244000s walltime 1.244486s
 *  Results for D = 100
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 3.438000s walltime 0.521576s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 2.644000s walltime 2.645543s
 *  Results for D = 200
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 5.860000s walltime 0.867629s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 5.092000s walltime 5.090811s
 *  Results for D = 300
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 8.564000s walltime 1.233921s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 7.770000s walltime 7.770405s
 *  Results for D = 400
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 10.974000s walltime 1.531718s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 10.126000s walltime 10.125524s
 *  Results for D = 500
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 13.558000s walltime 1.965116s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 12.894000s walltime 12.894182s
 *  -------------------------------------------------------------------------
 *  -------------------------------------------------------------------------
 *  Starting experiment for number of dimensions = 300, number of vectors = 100000, using kernel_width = 280.000000
 *  Results for D = 50
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 3.346000s walltime 0.580631s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 2.234000s walltime 2.234459s
 *  Results for D = 100
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 5.670000s walltime 0.878700s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 4.402000s walltime 4.401725s
 *  Results for D = 200
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 10.044000s walltime 1.441796s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 9.332000s walltime 9.332423s
 *  Results for D = 300
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 15.382000s walltime 2.138093s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 14.858000s walltime 14.858871s
 *  Results for D = 400
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 20.674000s walltime 2.905396s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 20.028000s walltime 20.030157s
 *  Results for D = 500
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 25.662000s walltime 3.550897s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 24.374000s walltime 24.374596s
 *  -------------------------------------------------------------------------
 *  -------------------------------------------------------------------------
 *  Starting experiment for number of dimensions = 600, number of vectors = 100000, using kernel_width = 580.000000
 *  Results for D = 50
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 4.414000s walltime 0.657778s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 3.490000s walltime 3.489634s
 *  Results for D = 100
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 8.456000s walltime 1.267112s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 7.458000s walltime 7.457174s
 *  Results for D = 200
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 16.922000s walltime 2.268248s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 16.142000s walltime 16.141996s
 *  Results for D = 300
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 25.584000s walltime 3.424675s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 25.752000s walltime 25.753305s
 *  Results for D = 400
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 34.392000s walltime 4.644195s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 34.340000s walltime 34.340004s
 *  Results for D = 500
 *  Time to process 5 x num=100000 dense_dot_range ops: cputime 44.028000s walltime 5.816031s
 *  Time to process 5 x num=100000 add_to_dense_vector ops: cputime 43.978000s walltime 43.979196s
 *  -------------------------------------------------------------------------
 *  Total time : 2531.890000s
 */
