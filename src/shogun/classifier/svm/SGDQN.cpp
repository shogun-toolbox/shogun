/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Shashwat Lal Das, Giovanni De Toni,
 *          Sergey Lisitsyn, Thoralf Klein, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/base/progress.h>
#include <shogun/classifier/svm/SGDQN.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/Signal.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

SGDQN::SGDQN()
: LinearMachine()
{
	init();
}

SGDQN::SGDQN(float64_t C)
: LinearMachine()
{
	init();

	C1=C;
	C2=C;
}


SGDQN::~SGDQN()
{

}

void SGDQN::set_loss_function(std::shared_ptr<LossFunction> loss_func)
{


	loss=std::move(loss_func);
}

void SGDQN::compute_ratio(float64_t* W,float64_t* W_1,float64_t* B,float64_t* dst,int32_t dim,float64_t lambda,float64_t loss_val)
{
	for (int32_t i=0; i < dim;i++)
	{
		float64_t diffw=W_1[i]-W[i];
		if(diffw)
			B[i]+=diffw/ (lambda*diffw+ loss_val*dst[i]);
		else
			B[i]+=1/lambda;
	}
}

void SGDQN::combine_and_clip(float64_t* Bc,float64_t* B,int32_t dim,float64_t c1,float64_t c2,float64_t v1,float64_t v2)
{
	for (int32_t i=0; i < dim;i++)
	{
		if(B[i])
		{
			Bc[i] = Bc[i] * c1 + B[i] * c2;
			Bc[i]= Math::min(Math::max(Bc[i],v1),v2);
		}
	}
}
bool SGDQN::train_machine(
    const std::shared_ptr<DotFeatures>& features, const std::shared_ptr<Labels>& labs)
{

	const auto binary_labels = labs->as<BinaryLabels>();

	int32_t num_train_labels = binary_labels->get_num_labels();
	int32_t num_vec = features->get_num_vectors();

	SGVector<float64_t> w(features->get_dim_feature_space());
	w.zero();

	float64_t lambda= 1.0/(C1*num_vec);

	// Shift t in order to have a
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	float64_t maxw = 1.0 / sqrt(lambda);
	float64_t typw = sqrt(maxw);
	float64_t eta0 = typw / Math::max(1.0,-loss->first_derivative(-typw,1));
	t = 1 / (eta0 * lambda);

	io::info("lambda={}, epochs={}, eta0={}", lambda, epochs, eta0);


	float64_t* Bc=SG_MALLOC(float64_t, w.vlen);
	SGVector<float64_t>::fill_vector(Bc, w.vlen, 1/lambda);

	float64_t* result=SG_MALLOC(float64_t, w.vlen);
	float64_t* B=SG_MALLOC(float64_t, w.vlen);

	//Calibrate
	calibrate(features);

	io::info("Training on {} vectors", num_vec);

	ELossType loss_type = loss->get_loss_type();
	bool is_log_loss = false;
	if ((loss_type == L_LOGLOSS) || (loss_type == L_LOGLOSSMARGIN))
		is_log_loss = true;

	for (auto e : SG_PROGRESS(range(epochs)))
	{
		COMPUTATION_CONTROLLERS
		count = skip;
		bool updateB=false;
		for (int32_t i=0; i<num_vec; i++)
		{
			SGVector<float64_t> v = features->get_computed_dot_feature_vector(i);
			ASSERT(w.vlen==v.vlen)
			float64_t eta = 1.0/t;
			float64_t y = binary_labels->get_label(i);
			float64_t z = y * features->dot(i, w);
			if(updateB==true)
			{
				if (z < 1 || is_log_loss)
				{
					SGVector<float64_t> w_1=w.clone();
					float64_t loss_1=-loss->first_derivative(z,1);
					SGVector<float64_t>::vector_multiply(result,Bc,v.vector,w.vlen);
					SGVector<float64_t>::add(w.vector,eta*loss_1*y,result,1.0,w.vector,w.vlen);
					float64_t z2 = y * features->dot(i, w);
					float64_t diffloss = -loss->first_derivative(z2,1) - loss_1;
					if(diffloss)
					{
						compute_ratio(w.vector,w_1.vector,B,v.vector,w.vlen,lambda,y*diffloss);
						if(t>skip)
							combine_and_clip(Bc,B,w.vlen,(t-skip)/(t+skip),2*skip/(t+skip),1/(100*lambda),100/lambda);
						else
							combine_and_clip(Bc,B,w.vlen,t/(t+skip),skip/(t+skip),1/(100*lambda),100/lambda);
					}
				}
				updateB=false;
			}
			else
			{
				if(--count<=0)
				{
					SGVector<float64_t>::vector_multiply(result,Bc,w.vector,w.vlen);
					SGVector<float64_t>::add(w.vector,-skip*lambda*eta,result,1.0,w.vector,w.vlen);
					count = skip;
					updateB=true;
				}

				if (z < 1 || is_log_loss)
				{
					SGVector<float64_t>::vector_multiply(result,Bc,v.vector,w.vlen);
					SGVector<float64_t>::add(w.vector,eta*-loss->first_derivative(z,1)*y,result,1.0,w.vector,w.vlen);
				}
			}
			t++;
		}
	}
	SG_FREE(result);
	SG_FREE(B);

	set_w(w);

	return true;
}

void SGDQN::calibrate(const std::shared_ptr<DotFeatures>& features)
{
	int32_t num_vec=features->get_num_vectors();
	int32_t c_dim=features->get_dim_feature_space();

	ASSERT(num_vec>0)
	ASSERT(c_dim>0)

	io::info("Estimating sparsity num_vec={} num_feat={}.", num_vec, c_dim);

	int32_t n = 0;
	float64_t r = 0;

	for (int32_t j=0; j<num_vec ; j++, n++)
		r += features->get_nnz_features_for_vector(j);


	// compute weight decay skip
	skip = (int32_t) ((16 * n * c_dim) / r);
}

void SGDQN::init()
{
	t=0;
	C1=1;
	C2=1;
	epochs=5;
	skip=1000;
	count=1000;

	loss=std::make_shared<HingeLoss>();


	SG_ADD(&C1, "C1", "Cost constant 1.", ParameterProperties::HYPER);
	SG_ADD(&C2, "C2", "Cost constant 2.", ParameterProperties::HYPER);
	SG_ADD(&epochs, "epochs", "epochs", ParameterProperties::HYPER);
	SG_ADD(&skip, "skip", "skip");
	SG_ADD(&count, "count", "count");
}
