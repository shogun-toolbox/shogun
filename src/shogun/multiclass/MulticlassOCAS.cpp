/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vojtech Franc, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/multiclass/MulticlassOCAS.h>

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/labels/MulticlassLabels.h>

#include <utility>

using namespace shogun;

struct mocas_data
{
	std::shared_ptr<DotFeatures> features;
	float64_t* W;
	float64_t* oldW;
	float64_t* full_A;
	float64_t* data_y;
	float64_t* output_values;
	uint32_t nY;
	uint32_t nData;
	uint32_t nDim;
	float64_t* new_a;
};

MulticlassOCAS::MulticlassOCAS() :
	LinearMulticlassMachine()
{
	register_parameters();
	set_C(1.0);
	set_epsilon(1e-2);
	set_max_iter(1000000);
	set_method(1);
	set_buf_size(5000);
}

MulticlassOCAS::MulticlassOCAS(float64_t C, const std::shared_ptr<Features>& train_features, std::shared_ptr<Labels> train_labels) :
	LinearMulticlassMachine(std::make_shared<MulticlassOneVsRestStrategy>(), train_features->as<DotFeatures>(), NULL, std::move(train_labels)), m_C(C)
{
	register_parameters();
	set_epsilon(1e-2);
	set_max_iter(1000000);
	set_method(1);
	set_buf_size(5000);
}

void MulticlassOCAS::register_parameters()
{
	SG_ADD(&m_C, "m_C", "regularization constant", ParameterProperties::HYPER);
	SG_ADD(&m_epsilon, "m_epsilon", "solver relative tolerance");
	SG_ADD(&m_max_iter, "m_max_iter", "max number of iterations");
	SG_ADD(&m_method, "m_method", "used solver method");
	SG_ADD(&m_buf_size, "m_buf_size", "buffer size");
}

MulticlassOCAS::~MulticlassOCAS()
{
}

bool MulticlassOCAS::train_machine(std::shared_ptr<Features> data)
{
	if (data)
		set_features(data->as<DotFeatures>());

	ASSERT(m_features)
	ASSERT(m_labels)
	ASSERT(m_multiclass_strategy)
	init_strategy();

	int32_t num_vectors = m_features->get_num_vectors();
	int32_t num_classes = m_multiclass_strategy->get_num_classes();
	int32_t num_features = m_features->get_dim_feature_space();

	float64_t C = m_C;
	SGVector<float64_t> labels = multiclass_labels(m_labels)->get_labels();
	uint32_t nY = num_classes;
	uint32_t nData = num_vectors;
	float64_t TolRel = m_epsilon;
	float64_t TolAbs = 0.0;
	float64_t QPBound = 0.0;
	float64_t MaxTime = m_max_train_time;
	uint32_t BufSize = m_buf_size;
	uint8_t Method = m_method;

	mocas_data user_data;
	user_data.features = m_features;
	user_data.W = SG_CALLOC(float64_t, (int64_t)num_features*num_classes);
	user_data.oldW = SG_CALLOC(float64_t, (int64_t)num_features*num_classes);
	user_data.new_a = SG_CALLOC(float64_t, (int64_t)num_features*num_classes);
	user_data.full_A = SG_CALLOC(float64_t, (int64_t)num_features*num_classes*m_buf_size);
	user_data.output_values = SG_CALLOC(float64_t, num_vectors);
	user_data.data_y = labels.vector;
	user_data.nY = num_classes;
	user_data.nDim = num_features;
	user_data.nData = num_vectors;

	ocas_return_value_T value =
	msvm_ocas_solver(C, labels.vector, nY, nData, TolRel, TolAbs,
	                 QPBound, MaxTime, BufSize, Method,
	                 &MulticlassOCAS::msvm_full_compute_W,
	                 &MulticlassOCAS::msvm_update_W,
	                 &MulticlassOCAS::msvm_full_add_new_cut,
	                 &MulticlassOCAS::msvm_full_compute_output,
	                 &MulticlassOCAS::msvm_sort_data,
	                 &MulticlassOCAS::msvm_print,
	                 &user_data);

	SG_DEBUG("Number of iterations [nIter] = {} ",value.nIter)
	SG_DEBUG("Number of cutting planes [nCutPlanes] = {} ",value.nCutPlanes)
	SG_DEBUG("Number of non-zero alphas [nNZAlpha] = {} ",value.nNZAlpha)
	SG_DEBUG("Number of training errors [trn_err] = {} ",value.trn_err)
	SG_DEBUG("Primal objective value [Q_P] = {} ",value.Q_P)
	SG_DEBUG("Dual objective value [Q_D] = {} ",value.Q_D)
	SG_DEBUG("Output time [output_time] = {} ",value.output_time)
	SG_DEBUG("Sort time [sort_time] = {} ",value.sort_time)
	SG_DEBUG("Add time [add_time] = {} ",value.add_time)
	SG_DEBUG("W time [w_time] = {} ",value.w_time)
	SG_DEBUG("QP solver time [qp_solver_time] = {} ",value.qp_solver_time)
	SG_DEBUG("OCAS time [ocas_time] = {} ",value.ocas_time)
	SG_DEBUG("Print time [print_time] = {} ",value.print_time)
	SG_DEBUG("QP exit flag [qp_exitflag] = {} ",value.qp_exitflag)
	SG_DEBUG("Exit flag [exitflag] = {} ",value.exitflag)

	m_machines.clear();
	for (int32_t i=0; i<num_classes; i++)
	{
		auto machine = std::make_shared<LinearMachine>();
		machine->set_w(SGVector<float64_t>(&user_data.W[i*num_features],num_features,false).clone());

		m_machines.push_back(machine);
	}

	SG_FREE(user_data.W);
	SG_FREE(user_data.oldW);
	SG_FREE(user_data.new_a);
	SG_FREE(user_data.full_A);
	SG_FREE(user_data.output_values);

	return true;
}

float64_t MulticlassOCAS::msvm_update_W(float64_t t, void* user_data)
{
	float64_t* oldW = ((mocas_data*)user_data)->oldW;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;
	SGVector<float64_t> W(((mocas_data*)user_data)->W, nDim*nY, false);

	for(uint32_t j=0; j < nY*nDim; j++)
		W[j] = oldW[j]*(1-t) + t*W[j];

	float64_t sq_norm_W = linalg::dot(W,W);

	return sq_norm_W;
}

void MulticlassOCAS::msvm_full_compute_W(float64_t *sq_norm_W, float64_t *dp_WoldW,
                                          float64_t *alpha, uint32_t nSel, void* user_data)
{
	float64_t* full_A = ((mocas_data*)user_data)->full_A;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;
	SGVector<float64_t> W(((mocas_data*)user_data)->W, nDim*nY, false);
	SGVector<float64_t> oldW(((mocas_data*)user_data)->oldW, nDim*nY, false);

	uint32_t i,j;

	sg_memcpy(oldW.vector, W.vector, sizeof(float64_t)*nDim*nY);
	linalg::zero(W);

	for(i=0; i<nSel; i++)
	{
		if(alpha[i] > 0)
		{
			for(j=0; j<nDim*nY; j++)
				W[j] += alpha[i]*full_A[LIBOCAS_INDEX(j,i,nDim*nY)];
		}
	}

	*sq_norm_W = linalg::dot(W,W);
	*dp_WoldW = linalg::dot(W,oldW);

	return;
}

int MulticlassOCAS::msvm_full_add_new_cut(float64_t *new_col_H, uint32_t *new_cut,
                                           uint32_t nSel, void* user_data)
{
	float64_t* full_A = ((mocas_data*)user_data)->full_A;
	float64_t* data_y = ((mocas_data*)user_data)->data_y;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;
	uint32_t nData = ((mocas_data*)user_data)->nData;
	SGVector<float64_t> new_a(((mocas_data*)user_data)->new_a, nDim*nY, false);
	auto features = ((mocas_data*)user_data)->features;

	float64_t sq_norm_a;
	uint32_t i, j, y, y2;

	linalg::zero(new_a);

	for(i=0; i < nData; i++)
	{
		y = (uint32_t)(data_y[i]);
		y2 = (uint32_t)new_cut[i];
		if(y2 != y)
		{
			features->add_to_dense_vec(1.0,i,&new_a[nDim*y],nDim);
			features->add_to_dense_vec(-1.0,i,&new_a[nDim*y2],nDim);
		}
	}

	// compute new_a'*new_a and insert new_a to the last column of full_A
	sq_norm_a = linalg::dot(new_a,new_a);
	for(j=0; j < nDim*nY; j++ )
		full_A[LIBOCAS_INDEX(j,nSel,nDim*nY)] = new_a[j];

	new_col_H[nSel] = sq_norm_a;
	for(i=0; i < nSel; i++)
	{
		float64_t tmp = 0;

		for(j=0; j < nDim*nY; j++ )
			tmp += new_a[j]*full_A[LIBOCAS_INDEX(j,i,nDim*nY)];

		new_col_H[i] = tmp;
	}

	return 0;
}

int MulticlassOCAS::msvm_full_compute_output(float64_t *output, void* user_data)
{
	float64_t* W = ((mocas_data*)user_data)->W;
	uint32_t nY = ((mocas_data*)user_data)->nY;
	uint32_t nDim = ((mocas_data*)user_data)->nDim;
	uint32_t nData = ((mocas_data*)user_data)->nData;
	float64_t* output_values = ((mocas_data*)user_data)->output_values;
	auto features = ((mocas_data*)user_data)->features;

	uint32_t i, y;

	for(y=0; y<nY; y++)
	{
		features->dense_dot_range(output_values,0,nData,NULL,&W[nDim*y],nDim,0.0);
		for (i=0; i<nData; i++)
			output[LIBOCAS_INDEX(y,i,nY)] = output_values[i];
	}

	return 0;
}

int MulticlassOCAS::msvm_sort_data(float64_t* vals, float64_t* data, uint32_t size)
{
	Math::qsort_index(vals, data, size);
	return 0;
}

void MulticlassOCAS::msvm_print(ocas_return_value_T value)
{
}

