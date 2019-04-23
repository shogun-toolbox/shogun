/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shashwat Lal Das, Soeren Sonnenburg, Giovanni De Toni, Sanuj Sharma,
 *          Thoralf Klein, Viktor Gal, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/classifier/svm/OnlineSVMSGD.h>
#include <shogun/lib/Signal.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

OnlineSVMSGD::OnlineSVMSGD()
: OnlineLinearMachine()
{
	init();
}

OnlineSVMSGD::OnlineSVMSGD(float64_t C)
: OnlineLinearMachine()
{
	init();

	C1=C;
	C2=C;
}

OnlineSVMSGD::OnlineSVMSGD(float64_t C, std::shared_ptr<StreamingDotFeatures> traindat)
: OnlineLinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
}

OnlineSVMSGD::~OnlineSVMSGD()
{

}

void OnlineSVMSGD::set_loss_function(std::shared_ptr<LossFunction> loss_func)
{


	loss=loss_func;
}

bool OnlineSVMSGD::train(std::shared_ptr<Features> data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			error("Specified features are not of type CStreamingDotFeatures");
		set_features(std::static_pointer_cast<StreamingDotFeatures>(data));
	}

	features->start_parser();

	// allocate memory for w and initialize everyting w and bias with 0
	ASSERT(features)
	ASSERT(features->get_has_labels())
	m_w = SGVector<float32_t>(1);
	bias=0;

	// Shift t in order to have a
	// reasonable initial learning rate.
	// This assumes |x| \approx 1.
	float64_t maxw = 1.0 / sqrt(lambda);
	float64_t typw = sqrt(maxw);
	float64_t eta0 = typw / Math::max(1.0,-loss->first_derivative(-typw,1));
	t = 1 / (eta0 * lambda);

	io::info("lambda={}, epochs={}, eta0={}", lambda, epochs, eta0);

	//do the sgd
	calibrate();
	if (features->is_seekable())
		features->reset_stream();

	ELossType loss_type = loss->get_loss_type();
	bool is_log_loss = false;
	if ((loss_type == L_LOGLOSS) || (loss_type == L_LOGLOSSMARGIN))
		is_log_loss = true;

	int32_t vec_count;
	for (auto e : SG_PROGRESS(range(epochs)))
	{
		COMPUTATION_CONTROLLERS
		vec_count=0;
		count = skip;
		while (features->get_next_example())
		{
			vec_count++;
			// Expand w vector if more features are seen in this example
			features->expand_if_required(m_w.vector, m_w.vlen);

			float64_t eta = 1.0 / (lambda * t);
			float64_t y = features->get_label();
			float64_t z = y * (features->dense_dot(m_w.vector, m_w.vlen) + bias);

			if (z < 1 || is_log_loss)
			{
				float64_t etd = -eta * loss->first_derivative(z,1);
				features->add_to_dense_vec(etd * y / wscale, m_w.vector, m_w.vlen);

				if (use_bias)
				{
					if (use_regularized_bias)
						bias *= 1 - eta * lambda * bscale;
					bias += etd * y * bscale;
				}
			}

			if (--count <= 0)
			{
				float32_t r = 1 - eta * lambda * skip;
				if (r < 0.8)
					r = pow(1 - eta * lambda, skip);
				linalg::scale(m_w, m_w, r);
				count = skip;
			}
			t++;

			features->release_example();
		}

		// If the stream is seekable, reset the stream to the first
		// example (for epochs > 1)
		if (features->is_seekable() && e < epochs-1)
			features->reset_stream();
		else
			break;

	}

	features->end_parser();
	float64_t wnorm = linalg::dot(m_w, m_w);
	io::info("Norm: {:.6f}, Bias: {:.6f}", wnorm, bias);

	return true;
}

void OnlineSVMSGD::calibrate(int32_t max_vec_num)
{
	int32_t c_dim=1;
	float32_t* c=SG_CALLOC(float32_t, c_dim);

	// compute average gradient size
	int32_t n = 0;
	float64_t m = 0;
	float64_t r = 0;

	while (features->get_next_example())
	{
		//Expand c if more features are seen in this example
		features->expand_if_required(c, c_dim);

		r += features->get_nnz_features_for_vector();
		features->add_to_dense_vec(1, c, c_dim, true);

		//waste cpu cycles for readability
		//(only changed dims need checking)
		m=Math::max(c, c_dim);
		n++;

		features->release_example();
		if (n>=max_vec_num || m > 1000)
			break;
	}

	io::print("Online SGD calibrated using {} vectors.\n", n);

	// bias update scaling
	bscale = 0.5*m/n;

	// compute weight decay skip
	skip = (int32_t) ((16 * n * c_dim) / r);

	io::info("using {} examples. skip={}  bscale={:.6f}", n, skip, bscale);

	SG_FREE(c);
}

void OnlineSVMSGD::init()
{
	t=1;
	C1=1;
	C2=1;
	lambda=1e-4;
	wscale=1;
	bscale=1;
	epochs=1;
	skip=1000;
	count=1000;
	use_bias=true;

	use_regularized_bias=false;

	loss=std::make_shared<HingeLoss>();


	SG_ADD(&C1, "C1", "Cost constant 1.", ParameterProperties::HYPER);
	SG_ADD(&C2, "C2", "Cost constant 2.", ParameterProperties::HYPER);
	SG_ADD(&lambda, "lambda", "Regularization parameter.", ParameterProperties::HYPER);
	SG_ADD(&wscale, "wscale", "W scale", ParameterProperties::HYPER);
	SG_ADD(&bscale, "bscale", "b scale", ParameterProperties::HYPER);
	SG_ADD(&epochs, "epochs", "epochs", ParameterProperties::HYPER);
	SG_ADD(&skip, "skip", "skip");
	SG_ADD(&count, "count", "count");
	SG_ADD(
	    &use_bias, "use_bias", "Indicates if bias is used.", ParameterProperties::SETTING);
	SG_ADD(
	    &use_regularized_bias, "use_regularized_bias",
	    "Indicates if bias is regularized.", ParameterProperties::SETTING);
}
