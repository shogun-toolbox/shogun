/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 pl8787
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
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

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/IndexFeatures.h>

using namespace shogun;

void test_custom_kernel_index_subsets()
{
	/* create some data */
	index_t m=10;
	index_t num_sub_row=3;
	index_t num_sub_col=2;

	Features* features=
			new DenseFeatures<float64_t>(DataGenerator::generate_mean_data(
			m, 2, 1));

	/* create a custom kernel */
	GaussianKernel* gaussian_kernel=new GaussianKernel(2,10);
	gaussian_kernel->init(features, features);
	CustomKernel* custom_kernel=new CustomKernel(gaussian_kernel);

	/* create random permutations */
	SGVector<index_t> row_subset(num_sub_row);
	SGVector<index_t> col_subset(num_sub_col);
	row_subset.range_fill();
	Math::permute(row_subset);
	col_subset.range_fill();
	Math::permute(col_subset);

	/* create index features */
	IndexFeatures* row_idx_feat=new IndexFeatures(row_subset);
	IndexFeatures* col_idx_feat=new IndexFeatures(col_subset);

	custom_kernel->init(row_idx_feat, col_idx_feat);

	SGMatrix<float64_t> gaussian_kernel_matrix=
			gaussian_kernel->get_kernel_matrix();

	SGMatrix<float64_t> custom_kernel_matrix=
			custom_kernel->get_kernel_matrix();

	custom_kernel_matrix.display_matrix("subset");

}

int main(int argc, char** argv)
{
	test_custom_kernel_index_subsets();

	return 0;
}


