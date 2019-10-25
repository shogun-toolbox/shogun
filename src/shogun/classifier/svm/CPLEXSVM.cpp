/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Fernando Iglesias, 
 *          Christian Widmer
 */

#include <shogun/classifier/svm/CPLEXSVM.h>
#include <shogun/lib/common.h>

#ifdef USE_CPLEX
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Cplex.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

CCPLEXSVM::CCPLEXSVM()
: SVM()
{
}

CCPLEXSVM::~CCPLEXSVM()
{
}

bool CCPLEXSVM::train_machine(Features* data)
{
	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_BINARY)

	bool result = false;
	CCplex cplex;

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
		{
			error("{}::train_machine(): Number of training vectors ({}) does"
					" not match number of labels ({})\n", get_name(),
					data->get_num_vectors(), m_labels->get_num_labels());
		}
		kernel->init(data, data);
	}

	if (cplex.init(E_QP))
	{
		int32_t n,m;
		int32_t num_label=0;
		SGVector<float64_t> y=((BinaryLabels*)m_labels)->get_labels();
		SGMatrix<float64_t> H=kernel->get_kernel_matrix();
		m=H.num_rows;
		n=H.num_cols;
		ASSERT(n>0 && n==m && n==num_label)
		float64_t* alphas=SG_MALLOC(float64_t, n);
		float64_t* lb=SG_MALLOC(float64_t, n);
		float64_t* ub=SG_MALLOC(float64_t, n);

		//hessian y'y.*K
		for (int32_t i=0; i<n; i++)
		{
			lb[i]=0;
			ub[i]=get_C1();

			for (int32_t j=0; j<n; j++)
				H[i*n+j]*=y[j]*y[i];
		}

		//feed qp to cplex


		int32_t j=0;
		for (int32_t i=0; i<n; i++)
		{
			if (alphas[i]>0)
			{
				//set_alpha(j, alphas[i]*labels->get_label(i)/etas[1]);
				set_alpha(j, alphas[i]*((BinaryLabels*) m_labels)->get_int_label(i));
				set_support_vector(j, i);
				j++;
			}
		}
		//compute_objective();
		io::info("obj = {:.16f}, rho = {:.16f}",get_objective(),get_bias());
		io::info("Number of SV: {}", get_num_support_vectors());

		SG_FREE(alphas);
		SG_FREE(lb);
		SG_FREE(ub);

		result = true;
	}

	if (!result)
		error("cplex svm failed");

	return result;
}
#endif
