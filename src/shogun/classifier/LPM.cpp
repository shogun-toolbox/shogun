/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Fernando Iglesias, 
 *          Christian Widmer
 */

#include <shogun/lib/config.h>

#ifdef USE_CPLEX

#include <shogun/classifier/LPM.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Cplex.h>

using namespace shogun;

CLPM::CLPM()
: LinearMachine(), C1(1), C2(1), use_bias(true), epsilon(1e-3)
{
}


CLPM::~CLPM()
{
}

bool CLPM::train_machine(Features* data)
{
	ASSERT(m_labels)
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type DotFeatures\n")
		set_features((DotFeatures*) data);
	}
	ASSERT(features)
	int32_t num_train_labels=m_labels->get_num_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels)

	w = SGVector<float64_t>(num_feat);

	int32_t num_params=1+2*num_feat+num_vec; //b,w+,w-,xi
	float64_t* params=SG_MALLOC(float64_t, num_params);
	memset(params,0,sizeof(float64_t)*num_params);

	CCplex solver;
	solver.init(E_LINEAR);
	SG_INFO("C=%f\n", C1)
	solver.setup_lpm(C1, (SparseFeatures<float64_t>*) features, (BinaryLabels*)m_labels, get_bias_enabled());
	if (get_max_train_time()>0)
		solver.set_time_limit(get_max_train_time());
	bool result=solver.optimize(params);
	solver.cleanup();

	set_bias(params[0]);
	for (int32_t i=0; i<num_feat; i++)
		w[i]=params[1+i]-params[1+num_feat+i];

//#define LPM_DEBUG
#ifdef LPM_DEBUG
	Math::display_vector(params,num_params, "params");
	SG_PRINT("bias=%f\n", bias)
	Math::display_vector(w,w_dim, "w");
	Math::display_vector(&params[1],w_dim, "w+");
	Math::display_vector(&params[1+w_dim],w_dim, "w-");
#endif
	SG_FREE(params);

	return result;
}
#endif
