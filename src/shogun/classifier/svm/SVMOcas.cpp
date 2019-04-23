/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vojtech Franc, Soeren Sonnenburg
 */

#include <shogun/classifier/svm/SVMOcas.h>

#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Time.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/Parallel.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

SVMOcas::SVMOcas()
: LinearMachine()
{
	init();
}

SVMOcas::SVMOcas(E_SVM_TYPE type)
: LinearMachine()
{
	init();
	method=type;
}

SVMOcas::SVMOcas(
	float64_t C, std::shared_ptr<Features> traindat, std::shared_ptr<Labels> trainlab)
: LinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(std::dynamic_pointer_cast<DotFeatures>(traindat));
	set_labels(trainlab);
}


SVMOcas::~SVMOcas()
{
}

bool SVMOcas::train_machine(std::shared_ptr<Features> data)
{
	SG_INFO("C=%f, epsilon=%f, bufsize=%d\n", get_C1(), get_epsilon(), bufsize)
	SG_DEBUG("use_bias = %i\n", get_bias_enabled())

	ASSERT(m_labels)
	ASSERT(m_labels->get_label_type() == LT_BINARY)
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type DotFeatures\n")
		set_features(std::static_pointer_cast<DotFeatures>(data));
	}
	ASSERT(features)

	int32_t num_vec=features->get_num_vectors();
	lab = SGVector<float64_t>(num_vec);
	auto labels = binary_labels(m_labels);
	for (int32_t i=0; i<num_vec; i++)
		lab[i] = labels->get_label(i);

	current_w = SGVector<float64_t>(features->get_dim_feature_space());
	current_w.zero();

	if (num_vec!=lab.vlen || num_vec<=0)
		SG_ERROR("num_vec=%d num_train_labels=%d\n", num_vec, lab.vlen)

	SG_FREE(old_w);
	old_w=SG_CALLOC(float64_t, current_w.vlen);
	bias=0;
	old_bias=0;

	tmp_a_buf=SG_CALLOC(float64_t, current_w.vlen);
	cp_value=SG_CALLOC(float64_t*, bufsize);
	cp_index=SG_CALLOC(uint32_t*, bufsize);
	cp_nz_dims=SG_CALLOC(uint32_t, bufsize);
	cp_bias=SG_CALLOC(float64_t, bufsize);

	float64_t TolAbs=0;
	float64_t QPBound=0;
	int32_t Method=0;
	if (method == SVM_OCAS)
		Method = 1;
	ocas_return_value_T result = svm_ocas_solver( get_C1(), num_vec, get_epsilon(),
			TolAbs, QPBound, get_max_train_time(), bufsize, Method,
			&SVMOcas::compute_W,
			&SVMOcas::update_W,
			&SVMOcas::add_new_cut,
			&SVMOcas::compute_output,
			&SVMOcas::sort,
			&SVMOcas::print,
			this);

	SG_INFO("Ocas Converged after %d iterations\n"
			"==================================\n"
			"timing statistics:\n"
			"output_time: %f s\n"
			"sort_time: %f s\n"
			"add_time: %f s\n"
			"w_time: %f s\n"
			"solver_time %f s\n"
			"ocas_time %f s\n\n", result.nIter, result.output_time, result.sort_time,
			result.add_time, result.w_time, result.qp_solver_time, result.ocas_time);

	SG_FREE(tmp_a_buf);

	primal_objective = result.Q_P;

	uint32_t num_cut_planes = result.nCutPlanes;

	SG_DEBUG("num_cut_planes=%d\n", num_cut_planes)
	for (uint32_t i=0; i<num_cut_planes; i++)
	{
		SG_DEBUG("cp_value[%d]=%p\n", i, cp_value)
		SG_FREE(cp_value[i]);
		SG_DEBUG("cp_index[%d]=%p\n", i, cp_index)
		SG_FREE(cp_index[i]);
	}

	SG_FREE(cp_value);
	cp_value=NULL;
	SG_FREE(cp_index);
	cp_index=NULL;
	SG_FREE(cp_nz_dims);
	cp_nz_dims=NULL;
	SG_FREE(cp_bias);
	cp_bias=NULL;

	SG_FREE(old_w);
	old_w=NULL;

	set_w(current_w);

	return true;
}

/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
float64_t SVMOcas::update_W( float64_t t, void* ptr )
{
  float64_t sq_norm_W = 0;
  auto o = (SVMOcas*)ptr;
  uint32_t nDim = (uint32_t) o->current_w.vlen;
  float64_t* W = o->current_w.vector;
  float64_t* oldW=o->old_w;

  for(uint32_t j=0; j <nDim; j++)
  {
	  W[j] = oldW[j]*(1-t) + t*W[j];
	  sq_norm_W += W[j]*W[j];
  }
  o->bias=o->old_bias*(1-t) + t*o->bias;
  sq_norm_W += Math::sq(o->bias);

  return( sq_norm_W );
}

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
int SVMOcas::add_new_cut(
	float64_t *new_col_H, uint32_t *new_cut, uint32_t cut_length,
	uint32_t nSel, void* ptr)
{
	auto o = (SVMOcas*)ptr;
	auto f = o->features;
	uint32_t nDim=(uint32_t) o->current_w.vlen;
	float64_t* y = o->lab.vector;

	float64_t** c_val = o->cp_value;
	uint32_t** c_idx = o->cp_index;
	uint32_t* c_nzd = o->cp_nz_dims;
	float64_t* c_bias = o->cp_bias;

	float64_t sq_norm_a;
	uint32_t i, j, nz_dims;

	/* temporary vector */
	float64_t* new_a = o->tmp_a_buf;
	memset(new_a, 0, sizeof(float64_t)*nDim);

	for(i=0; i < cut_length; i++)
	{
		f->add_to_dense_vec(y[new_cut[i]], new_cut[i], new_a, nDim);

		if (o->use_bias)
			c_bias[nSel]+=y[new_cut[i]];
	}

	/* compute new_a'*new_a and count number of non-zerou dimensions */
	nz_dims = 0;
	sq_norm_a = Math::sq(c_bias[nSel]);
	for(j=0; j < nDim; j++ ) {
		if(new_a[j] != 0) {
			nz_dims++;
			sq_norm_a += new_a[j]*new_a[j];
		}
	}

	/* sparsify new_a and insert it to the last column of sparse_A */
	c_nzd[nSel] = nz_dims;
	c_idx[nSel]=NULL;
	c_val[nSel]=NULL;

	if(nz_dims > 0)
	{
		c_idx[nSel]=SG_MALLOC(uint32_t, nz_dims);
		c_val[nSel]=SG_MALLOC(float64_t, nz_dims);

		uint32_t idx=0;
		for(j=0; j < nDim; j++ )
		{
			if(new_a[j] != 0)
			{
				c_idx[nSel][idx] = j;
				c_val[nSel][idx++] = new_a[j];
			}
		}
	}

	new_col_H[nSel] = sq_norm_a;

	for(i=0; i < nSel; i++)
	{
		float64_t tmp = c_bias[nSel]*c_bias[i];
		for(j=0; j < c_nzd[i]; j++)
			tmp += new_a[c_idx[i][j]]*c_val[i][j];

		new_col_H[i] = tmp;
	}
	//Math::display_vector(new_col_H, nSel+1, "new_col_H");
	//Math::display_vector((int32_t*) c_idx[nSel], (int32_t) nz_dims, "c_idx");
	//Math::display_vector((float64_t*) c_val[nSel], nz_dims, "c_val");
	return 0;
}

int SVMOcas::sort(float64_t* vals, float64_t* data, uint32_t size)
{
	Math::qsort_index(vals, data, size);
	return 0;
}

/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
int SVMOcas::compute_output(float64_t *output, void* ptr)
{
	auto o = (SVMOcas*)ptr;
	auto f=o->features;
	int32_t nData=f->get_num_vectors();

	float64_t* y = o->lab.vector;

	f->dense_dot_range(output, 0, nData, y, o->current_w.vector, o->current_w.vlen, 0.0);

	for (int32_t i=0; i<nData; i++)
		output[i]+=y[i]*o->bias;
	return 0;
}

/*----------------------------------------------------------------------
  sq_norm_W = compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void SVMOcas::compute_W(
	float64_t *sq_norm_W, float64_t *dp_WoldW, float64_t *alpha, uint32_t nSel,
	void* ptr )
{
	auto o = (SVMOcas*)ptr;
	uint32_t nDim= (uint32_t) o->current_w.vlen;
	Math::swap(o->current_w.vector, o->old_w);
	SGVector<float64_t> W(o->current_w.vector, nDim, false);
	SGVector<float64_t> oldW(o->old_w, nDim, false);
	linalg::zero(W);
	float64_t old_bias=o->bias;
	float64_t bias=0;

	float64_t** c_val = o->cp_value;
	uint32_t** c_idx = o->cp_index;
	uint32_t* c_nzd = o->cp_nz_dims;
	float64_t* c_bias = o->cp_bias;

	for(uint32_t i=0; i<nSel; i++)
	{
		uint32_t nz_dims = c_nzd[i];

		if(nz_dims > 0 && alpha[i] > 0)
		{
			for(uint32_t j=0; j < nz_dims; j++)
				W[c_idx[i][j]] += alpha[i]*c_val[i][j];
		}
		bias += c_bias[i]*alpha[i];
	}

	*sq_norm_W = linalg::dot(W, W) + Math::sq(bias);
	*dp_WoldW = linalg::dot(W, oldW) + bias*old_bias;
	//SG_PRINT("nSel=%d sq_norm_W=%f dp_WoldW=%f\n", nSel, *sq_norm_W, *dp_WoldW)

	o->bias = bias;
	o->old_bias = old_bias;
}

void SVMOcas::init()
{
	use_bias = true;
	bufsize = 3000;
	C1 = 1;
	C2 = 1;

	epsilon = 1e-3;
	method = SVM_OCAS;
	old_w = NULL;
	tmp_a_buf = NULL;
	cp_value = NULL;
	cp_index = NULL;
	cp_nz_dims = NULL;
	cp_bias = NULL;

	primal_objective = 0.0;

	SG_ADD(&C1, "C1", "Cost constant 1.", ParameterProperties::HYPER);
	SG_ADD(&C2, "C2", "Cost constant 2.", ParameterProperties::HYPER);
	SG_ADD(&use_bias, "use_bias", "Indicates if bias is used.");
	SG_ADD(&epsilon, "epsilon", "Convergence precision.");
	SG_ADD(&bufsize, "bufsize", "Maximum number of cutting planes.");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&method, "method", "SVMOcas solver type.",
	    ParameterProperties::NONE, SG_OPTIONS(SVM_OCAS, SVM_BMRM));
}

float64_t SVMOcas::compute_primal_objective() const
{
	return primal_objective;
}
