/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vojtech Franc, Soeren Sonnenburg
 */

#include <shogun/classifier/svm/WDSVMOcas.h>

#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/Time.h>
#include <shogun/base/Parallel.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/external/libocas.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/Alphabet.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>

#include <utility>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct wdocas_thread_params_output
{
	float32_t* out;
	int32_t* val;
	float64_t* output;
	WDSVMOcas* wdocas;
	int32_t start;
	int32_t end;
};

struct wdocas_thread_params_add
{
	WDSVMOcas* wdocas;
	float32_t* new_a;
	uint32_t* new_cut;
	int32_t start;
	int32_t end;
	uint32_t cut_length;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

WDSVMOcas::WDSVMOcas()
: Machine(), use_bias(false), bufsize(3000), C1(1), C2(1),
	epsilon(1e-3), method(SVM_OCAS)
{
	unstable(SOURCE_LOCATION);

	w=NULL;
	old_w=NULL;
	features=NULL;
	degree=6;
	from_degree=40;
	wd_weights=NULL;
	w_offsets=NULL;
	normalization_const=1.0;
}

WDSVMOcas::WDSVMOcas(E_SVM_TYPE type)
: Machine(), use_bias(false), bufsize(3000), C1(1), C2(1),
	epsilon(1e-3), method(type)
{
	w=NULL;
	old_w=NULL;
	features=NULL;
	degree=6;
	from_degree=40;
	wd_weights=NULL;
	w_offsets=NULL;
	normalization_const=1.0;
}

WDSVMOcas::WDSVMOcas(
	float64_t C, int32_t d, int32_t from_d, std::shared_ptr<StringFeatures<uint8_t>> traindat)
: Machine(), use_bias(false), bufsize(3000), C1(C), C2(C), epsilon(1e-3),
	degree(d), from_degree(from_d)
{
	w=NULL;
	old_w=NULL;
	method=SVM_OCAS;
	features=std::move(traindat);
	wd_weights=NULL;
	w_offsets=NULL;
	normalization_const=1.0;
}


WDSVMOcas::~WDSVMOcas()
{
}

std::shared_ptr<BinaryLabels> WDSVMOcas::apply_binary(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<BinaryLabels>(outputs);
}

std::shared_ptr<RegressionLabels> WDSVMOcas::apply_regression(std::shared_ptr<Features> data)
{
	SGVector<float64_t> outputs = apply_get_outputs(data);
	return std::make_shared<RegressionLabels>(outputs);
}

SGVector<float64_t> WDSVMOcas::apply_get_outputs(const std::shared_ptr<Features>& data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_BYTE)
		{
			error("Features not of class string type byte");
		}

		set_features(std::static_pointer_cast<StringFeatures<uint8_t>>(data));
	}
	ASSERT(features)

	set_wd_weights();
	set_normalization_const();

	SGVector<float64_t> outputs;
	if (features)
	{
		int32_t num=features->get_num_vectors();
		ASSERT(num>0)

		outputs = SGVector<float64_t>(num);

		for (int32_t i=0; i<num; i++)
			outputs[i] = apply_one(i);
	}

	return outputs;
}

int32_t WDSVMOcas::set_wd_weights()
{
	ASSERT(degree>0 && degree<=8)
	SG_FREE(wd_weights);
	wd_weights=SG_MALLOC(float32_t, degree);
	SG_FREE(w_offsets);
	w_offsets=SG_MALLOC(int32_t, degree);
	int32_t w_dim_single_c=0;

	for (int32_t i=0; i<degree; i++)
	{
		w_offsets[i]=Math::pow(alphabet_size, i+1);
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
		w_dim_single_c+=w_offsets[i];
	}
	return w_dim_single_c;
}

bool WDSVMOcas::train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs)
{
	io::info("C={}, epsilon={}, bufsize={}", get_C1(), get_epsilon(), bufsize);

	if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_BYTE)
	{
		error("Features not of class string type byte");
	}

	ASSERT(get_features())
	features = data->as<StringFeatures<uint8_t>>();
	auto alphabet=features->get_alphabet();
	ASSERT(alphabet && alphabet->get_alphabet()==RAWDNA)

	alphabet_size=alphabet->get_num_symbols();
	string_length=features->get_num_vectors();
	SGVector<float64_t> labvec=(std::static_pointer_cast<BinaryLabels>(labs))->get_labels();
	lab=labvec.vector;

	w_dim_single_char=set_wd_weights();
	//Math::display_vector(wd_weights, degree, "wd_weights");
	SG_DEBUG("w_dim_single_char={}", w_dim_single_char)
	w_dim=string_length*w_dim_single_char;
	SG_DEBUG("cutting plane has {} dims", w_dim)
	num_vec=features->get_max_vector_length();

	set_normalization_const();
	io::info("num_vec: {} num_lab: {}", num_vec, labvec.vlen);
	ASSERT(num_vec==labvec.vlen)
	ASSERT(num_vec>0)

	SG_FREE(w);
	w=SG_MALLOC(float32_t, w_dim);
	memset(w, 0, w_dim*sizeof(float32_t));

	SG_FREE(old_w);
	old_w=SG_MALLOC(float32_t, w_dim);
	memset(old_w, 0, w_dim*sizeof(float32_t));
	bias=0;
	old_bias=0;

	cuts=SG_MALLOC(float32_t*, bufsize);
	memset(cuts, 0, sizeof(*cuts)*bufsize);
	cp_bias=SG_MALLOC(float64_t, bufsize);
	memset(cp_bias, 0, sizeof(float64_t)*bufsize);

/////speed tests/////
	/*float64_t* tmp = SG_MALLOC(float64_t, num_vec);
	float64_t start=Time::get_curtime();
	Math::random_vector(w, w_dim, (float32_t) 0, (float32_t) 1000);
	compute_output(tmp, this);
	start=Time::get_curtime()-start;
	io::print("timing:{}\n", start);
	SG_FREE(tmp);
	exit(1);*/
/////speed tests/////
	float64_t TolAbs=0;
	float64_t QPBound=0;
	uint8_t Method=0;
	if (method == SVM_OCAS)
		Method = 1;
	ocas_return_value_T result = svm_ocas_solver( get_C1(), num_vec, get_epsilon(),
			TolAbs, QPBound, get_max_train_time(), bufsize, Method,
			&WDSVMOcas::compute_W,
			&WDSVMOcas::update_W,
			&WDSVMOcas::add_new_cut,
			&WDSVMOcas::compute_output,
			&WDSVMOcas::sort,
			&WDSVMOcas::print,
			this);

	io::info("Ocas Converged after {} iterations"
			"==================================\n"
			"timing statistics:\n"
			"output_time: {} s\n"
			"sort_time: {} s\n"
			"add_time: {} s\n"
			"w_time: {} s\n"
			"solver_time {} s\n"
			"ocas_time {} s", result.nIter, result.output_time, result.sort_time,
			result.add_time, result.w_time, result.qp_solver_time, result.ocas_time);

	for (int32_t i=bufsize-1; i>=0; i--)
		SG_FREE(cuts[i]);
	SG_FREE(cuts);

	lab=NULL;


	return true;
}

/*----------------------------------------------------------------------------------
  sq_norm_W = sparse_update_W( t ) does the following:

  W = oldW*(1-t) + t*W;
  sq_norm_W = W'*W;

  ---------------------------------------------------------------------------------*/
float64_t WDSVMOcas::update_W( float64_t t, void* ptr )
{
  float64_t sq_norm_W = 0;
  auto o = (WDSVMOcas*)ptr;
  uint32_t nDim = (uint32_t) o->w_dim;
  float32_t* W=o->w;
  float32_t* oldW=o->old_w;
  float64_t bias=o->bias;
  float64_t old_bias=bias;

  for(uint32_t j=0; j <nDim; j++)
  {
	  W[j] = oldW[j]*(1-t) + t*W[j];
	  sq_norm_W += W[j]*W[j];
  }

  bias=old_bias*(1-t) + t*bias;
  sq_norm_W += Math::sq(bias);

  o->bias=bias;
  o->old_bias=old_bias;

  return( sq_norm_W );
}

/*----------------------------------------------------------------------------------
  sparse_add_new_cut( new_col_H, new_cut, cut_length, nSel ) does the following:

    new_a = sum(data_X(:,find(new_cut ~=0 )),2);
    new_col_H = [sparse_A(:,1:nSel)'*new_a ; new_a'*new_a];
    sparse_A(:,nSel+1) = new_a;

  ---------------------------------------------------------------------------------*/
void* WDSVMOcas::add_new_cut_helper( void* ptr)
{
	wdocas_thread_params_add* p = (wdocas_thread_params_add*) ptr;
	auto o = p->wdocas;
	int32_t start = p->start;
	int32_t end = p->end;
	int32_t string_length = o->string_length;
	//uint32_t nDim=(uint32_t) o->w_dim;
	uint32_t cut_length=p->cut_length;
	uint32_t* new_cut=p->new_cut;
	int32_t* w_offsets = o->w_offsets;
	float64_t* y = o->lab;
	int32_t alphabet_size = o->alphabet_size;
	float32_t* wd_weights = o->wd_weights;
	int32_t degree = o->degree;
	auto f = o->features;
	float64_t normalization_const = o->normalization_const;

	// temporary vector
	float32_t* new_a = p->new_a;
	//float32_t* new_a = SG_MALLOC(float32_t, nDim);
	//memset(new_a, 0, sizeof(float32_t)*nDim);

	int32_t* val=SG_MALLOC(int32_t, cut_length);
	for (int32_t j=start; j<end; j++)
	{
		int32_t offs=o->w_dim_single_char*j;
		memset(val,0,sizeof(int32_t)*cut_length);
		int32_t lim=Math::min(degree, string_length-j);
		int32_t len;

		for (int32_t k=0; k<lim; k++)
		{
			bool free_vec;
			uint8_t* vec = f->get_feature_vector(j+k, len, free_vec);
			float32_t wd = wd_weights[k]/normalization_const;

			for(uint32_t i=0; i < cut_length; i++)
			{
				val[i]=val[i]*alphabet_size + vec[new_cut[i]];
				new_a[offs+val[i]]+=wd * y[new_cut[i]];
			}
			offs+=w_offsets[k];
			f->free_feature_vector(vec, j+k, free_vec);
		}
	}

	//p->new_a=new_a;
	SG_FREE(val);
	return NULL;
}

int WDSVMOcas::add_new_cut(
	float64_t *new_col_H, uint32_t *new_cut, uint32_t cut_length,
	uint32_t nSel, void* ptr)
{
	auto o = (WDSVMOcas*)ptr;
	uint32_t i;
	float64_t* c_bias = o->cp_bias;
	uint32_t nDim=(uint32_t) o->w_dim;
	float32_t** cuts=o->cuts;
	SGVector<float32_t> new_a(nDim);
#ifdef HAVE_PTHREAD

	wdocas_thread_params_add* params_add=SG_MALLOC(wdocas_thread_params_add, env()->get_num_threads());
	pthread_t* threads=SG_MALLOC(pthread_t, env()->get_num_threads());

	int32_t string_length = o->string_length;
	int32_t t;
	int32_t nthreads=env()->get_num_threads()-1;
	int32_t step= string_length/env()->get_num_threads();

	if (step<1)
	{
		nthreads=string_length-1;
		step=1;
	}

	for (t=0; t<nthreads; t++)
	{
		params_add[t].wdocas=o;
		//params_add[t].new_a=NULL;
		params_add[t].new_a=new_a.vector;
		params_add[t].new_cut=new_cut;
		params_add[t].start = step*t;
		params_add[t].end = step*(t+1);
		params_add[t].cut_length = cut_length;

		if (pthread_create(&threads[t], NULL, &WDSVMOcas::add_new_cut_helper, (void*)&params_add[t]) != 0)
		{
			nthreads=t;
			io::warn("thread creation failed");
			break;
		}
	}

	params_add[t].wdocas=o;
	//params_add[t].new_a=NULL;
	params_add[t].new_a=new_a.vector;
	params_add[t].new_cut=new_cut;
	params_add[t].start = step*t;
	params_add[t].end = string_length;
	params_add[t].cut_length = cut_length;
	add_new_cut_helper(&params_add[t]);
	//float32_t* new_a=params_add[t].new_a;

	for (t=0; t<nthreads; t++)
	{
		if (pthread_join(threads[t], NULL) != 0)
			io::warn("pthread_join failed");

		//float32_t* a=params_add[t].new_a;
		//for (i=0; i<nDim; i++)
		//	new_a[i]+=a[i];
		//SG_FREE(a);
	}
	SG_FREE(threads);
	SG_FREE(params_add);
#endif /* HAVE_PTHREAD */
	for(i=0; i < cut_length; i++)
	{
		if (o->use_bias)
			c_bias[nSel]+=o->lab[new_cut[i]];
	}

	// insert new_a into the last column of sparse_A
	for(i=0; i < nSel; i++)
	{
		SGVector<float32_t> cut_wrap(cuts[i], nDim, false);
		new_col_H[i] = linalg::dot(new_a, cut_wrap) + c_bias[nSel]*c_bias[i];
	}
	new_col_H[nSel] = linalg::dot(new_a, new_a) + Math::sq(c_bias[nSel]);

	cuts[nSel]=new_a;
	//Math::display_vector(new_col_H, nSel+1, "new_col_H");
	//Math::display_vector(cuts[nSel], nDim, "cut[nSel]");
	//

	return 0;
}

int WDSVMOcas::sort( float64_t* vals, float64_t* data, uint32_t size)
{
	Math::qsort_index(vals, data, size);
	return 0;
}

/*----------------------------------------------------------------------
  sparse_compute_output( output ) does the follwing:

  output = data_X'*W;
  ----------------------------------------------------------------------*/
void* WDSVMOcas::compute_output_helper(void* ptr)
{
	wdocas_thread_params_output* p = (wdocas_thread_params_output*) ptr;
	WDSVMOcas* o = p->wdocas;
	int32_t start = p->start;
	int32_t end = p->end;
	float32_t* out = p->out;
	float64_t* output = p->output;
	int32_t* val = p->val;

	auto f=o->get_features();

	int32_t degree = o->degree;
	int32_t string_length = o->string_length;
	int32_t alphabet_size = o->alphabet_size;
	int32_t* w_offsets = o->w_offsets;
	float32_t* wd_weights = o->wd_weights;
	float32_t* w= o->w;

	float64_t* y = o->lab;
	float64_t normalization_const = o->normalization_const;


	for (int32_t j=0; j<string_length; j++)
	{
		int32_t offs=o->w_dim_single_char*j;
		for (int32_t i=start ; i<end; i++)
			val[i]=0;

		int32_t lim=Math::min(degree, string_length-j);
		int32_t len;

		for (int32_t k=0; k<lim; k++)
		{
			bool free_vec;
			uint8_t* vec=f->get_feature_vector(j+k, len, free_vec);
			float32_t wd = wd_weights[k];

			for (int32_t i=start; i<end; i++) // quite fast 1.9s
			{
				val[i]=val[i]*alphabet_size + vec[i];
				out[i]+=wd*w[offs+val[i]];
			}

			/*for (int32_t i=0; i<nData/4; i++) // slowest 2s
			{
				uint32_t x=((uint32_t*) vec)[i];
				int32_t ii=4*i;
				val[ii]=val[ii]*alphabet_size + (x&255);
				val[ii+1]=val[ii+1]*alphabet_size + ((x>>8)&255);
				val[ii+2]=val[ii+2]*alphabet_size + ((x>>16)&255);
				val[ii+3]=val[ii+3]*alphabet_size + (x>>24);
				out[ii]+=wd*w[offs+val[ii]];
				out[ii+1]+=wd*w[offs+val[ii+1]];
				out[ii+2]+=wd*w[offs+val[ii+2]];
				out[ii+3]+=wd*w[offs+val[ii+3]];
			}*/

			/*for (int32_t i=0; i<nData>>3; i++) // fastest on 64bit: 1.5s
			{
				uint64_t x=((uint64_t*) vec)[i];
				int32_t ii=i<<3;
				val[ii]=val[ii]*alphabet_size + (x&255);
				val[ii+1]=val[ii+1]*alphabet_size + ((x>>8)&255);
				val[ii+2]=val[ii+2]*alphabet_size + ((x>>16)&255);
				val[ii+3]=val[ii+3]*alphabet_size + ((x>>24)&255);
				val[ii+4]=val[ii+4]*alphabet_size + ((x>>32)&255);
				val[ii+5]=val[ii+5]*alphabet_size + ((x>>40)&255);
				val[ii+6]=val[ii+6]*alphabet_size + ((x>>48)&255);
				val[ii+7]=val[ii+7]*alphabet_size + (x>>56);
				out[ii]+=wd*w[offs+val[ii]];
				out[ii+1]+=wd*w[offs+val[ii+1]];
				out[ii+2]+=wd*w[offs+val[ii+2]];
				out[ii+3]+=wd*w[offs+val[ii+3]];
				out[ii+4]+=wd*w[offs+val[ii+4]];
				out[ii+5]+=wd*w[offs+val[ii+5]];
				out[ii+6]+=wd*w[offs+val[ii+6]];
				out[ii+7]+=wd*w[offs+val[ii+7]];
			}*/
			offs+=w_offsets[k];
			f->free_feature_vector(vec, j+k, free_vec);
		}
	}

	for (int32_t i=start; i<end; i++)
		output[i]=y[i]*o->bias + out[i]*y[i]/normalization_const;

	//Math::display_vector(o->w, o->w_dim, "w");
	//Math::display_vector(output, nData, "out");
	return NULL;
}

int WDSVMOcas::compute_output( float64_t *output, void* ptr )
{
#ifdef HAVE_PTHREAD
	auto o = (WDSVMOcas*)ptr;
	int32_t nData=o->num_vec;
	wdocas_thread_params_output* params_output=SG_MALLOC(wdocas_thread_params_output, env()->get_num_threads());
	pthread_t* threads = SG_MALLOC(pthread_t, env()->get_num_threads());

	float32_t* out=SG_MALLOC(float32_t, nData);
	int32_t* val=SG_MALLOC(int32_t, nData);
	memset(out, 0, sizeof(float32_t)*nData);

	int32_t t;
	int32_t nthreads=env()->get_num_threads()-1;
	int32_t step= nData/env()->get_num_threads();

	if (step<1)
	{
		nthreads=nData-1;
		step=1;
	}

	for (t=0; t<nthreads; t++)
	{
		params_output[t].wdocas=o;
		params_output[t].output=output;
		params_output[t].out=out;
		params_output[t].val=val;
		params_output[t].start = step*t;
		params_output[t].end = step*(t+1);

		//io::print("t={} start={} end={} output={}\n", t, params_output[t].start, params_output[t].end, fmt::ptr(params_output[t].output));
		if (pthread_create(&threads[t], NULL, &WDSVMOcas::compute_output_helper, (void*)&params_output[t]) != 0)
		{
			nthreads=t;
			io::warn("thread creation failed");
			break;
		}
	}

	params_output[t].wdocas=o;
	params_output[t].output=output;
	params_output[t].out=out;
	params_output[t].val=val;
	params_output[t].start = step*t;
	params_output[t].end = nData;
	compute_output_helper(&params_output[t]);
	//io::print("t={} start={} end={} output={}\n", t, params_output[t].start, params_output[t].end, fmt::ptr(params_output[t].output));

	for (t=0; t<nthreads; t++)
	{
		if (pthread_join(threads[t], NULL) != 0)
			io::warn("pthread_join failed");
	}
	SG_FREE(threads);
	SG_FREE(params_output);
	SG_FREE(val);
	SG_FREE(out);
#endif /* HAVE_PTHREAD */
	return 0;
}
/*----------------------------------------------------------------------
  sq_norm_W = compute_W( alpha, nSel ) does the following:

  oldW = W;
  W = sparse_A(:,1:nSel)'*alpha;
  sq_norm_W = W'*W;
  dp_WoldW = W'*oldW';

  ----------------------------------------------------------------------*/
void WDSVMOcas::compute_W(
	float64_t *sq_norm_W, float64_t *dp_WoldW, float64_t *alpha, uint32_t nSel,
	void* ptr)
{
	auto o = (WDSVMOcas*)ptr;
	uint32_t nDim= (uint32_t) o->w_dim;
	Math::swap(o->w, o->old_w);
	SGVector<float32_t> W(o->w, nDim, false);
	linalg::zero(W);
	SGVector<float32_t> oldW(o->old_w, nDim, false);
	float32_t** cuts=o->cuts;
	float64_t* c_bias = o->cp_bias;
	float64_t old_bias=o->bias;
	float64_t bias=0;

	for (uint32_t i=0; i<nSel; i++)
	{
		if (alpha[i] > 0)
			SGVector<float32_t>::vec1_plus_scalar_times_vec2(W.vector, (float32_t) alpha[i], cuts[i], nDim);

		bias += c_bias[i]*alpha[i];
	}

	*sq_norm_W = linalg::dot(W, W) +Math::sq(bias);
	*dp_WoldW = linalg::dot(W, oldW) + bias*old_bias;;
	//io::print("nSel={} sq_norm_W={} dp_WoldW={}\n", nSel, *sq_norm_W, *dp_WoldW);

	o->bias = bias;
	o->old_bias = old_bias;
}
