/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/Time.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/eigen3.h>
#include <map>

using namespace shogun;
using namespace Eigen;

std::pair<float64_t,float64_t> test()
{
	CTime *time=new CTime();

	const index_t n=1000;
	const index_t d=3;
	SGMatrix<float64_t> data_p(d, n);
	Map<MatrixXd> data_pm(data_p.matrix, data_p.num_rows, data_p.num_cols);
	data_pm=MatrixXd::Random(d, n);
	SGMatrix<float64_t> data_q(d, n);
	Map<MatrixXd> data_qm(data_q.matrix, data_q.num_rows, data_q.num_cols);
	data_qm=MatrixXd::Random(d, n);

	CDenseFeatures<float64_t>* feats_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* feats_q=new CDenseFeatures<float64_t>(data_q);
	CGaussianKernel* kernel=new CGaussianKernel(feats_p, feats_q, 2);
	CCustomKernel* precomputed_kernel=new CCustomKernel(kernel);

	// BENCHMARK_1
	time->start();
	float64_t sum1=precomputed_kernel->sum_block(0, 0, n, n);
	float64_t time1=time->cur_time_diff();

	float64_t sum2=0.0;
	SGMatrix<float64_t> km=precomputed_kernel->get_kernel_matrix();
	Map<MatrixXd> k_m(km.matrix, km.num_rows, km.num_cols);

	// BENCHMARK_2
	time->start();
	sum2=k_m.sum();
	float64_t time2=time->cur_time_diff();

	ASSERT(CMath::abs(sum1-sum2) <= 1E-5);

	SG_UNREF(kernel);
	SG_UNREF(precomputed_kernel);
	SG_UNREF(time);

	return std::make_pair(time1, time2);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();
	//sg_io->set_loglevel(MSG_DEBUG);
	//sg_io->set_location_info(MSG_FUNCTION);
	float64_t time1=0.0, time2=0.0;
	float64_t var1=0.0, var2=0.0;
	index_t num_runs=100;
	for (index_t i=1; i<=num_runs; ++i)
	{
		std::pair<float64_t,float64_t> time=test();
		float64_t delta=time.first - time1;
		time1+=delta/i;
		var1+=delta*(time.first - time1);
		delta=time.second - time2;
		time2+=delta/i;
		var2+=delta*(time.second - time2);
	}
	var1/=num_runs;
	var2/=num_runs;
	SG_SPRINT("mean %f\t var %f\n", time1, var1);
	SG_SPRINT("mean %f\t var %f\n", time2, var2);
	exit_shogun();
	return 0;
}

