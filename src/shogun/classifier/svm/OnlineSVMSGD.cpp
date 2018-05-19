/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shashwat Lal Das, Soeren Sonnenburg, Giovanni De Toni, Sanuj Sharma,
 *          Thoralf Klein, Viktor Gal, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/classifier/svm/OnlineSVMSGD.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/base/Parameter.h>
#include <shogun/base/progress.h>
#include <shogun/lib/Signal.h>
#include <shogun/loss/HingeLoss.h>

using namespace shogun;

COnlineSVMSGD::COnlineSVMSGD()
: COnlineLinearMachine()
{
	init();
}

COnlineSVMSGD::COnlineSVMSGD(float64_t C)
: COnlineLinearMachine()
{
	init();

	C1=C;
	C2=C;
}

COnlineSVMSGD::COnlineSVMSGD(float64_t C, CStreamingDotFeatures* traindat)
: COnlineLinearMachine()
{
	init();
	C1=C;
	C2=C;

	set_features(traindat);
}

COnlineSVMSGD::~COnlineSVMSGD()
{
	SG_UNREF(loss);
}

void COnlineSVMSGD::set_loss_function(CLossFunction* loss_func)
{
	SG_REF(loss_func);
	SG_UNREF(loss);
	loss=loss_func;
}

bool COnlineSVMSGD::train(CFeatures* data)
{
	if (data)
	{
		if (!data->has_property(FP_STREAMING_DOT))
			SG_ERROR("Specified features are not of type CStreamingDotFeatures\n")
		set_features((CStreamingDotFeatures*) data);
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
	float64_t eta0 = typw / CMath::max(1.0,-loss->first_derivative(-typw,1));
	t = 1 / (eta0 * lambda);

	SG_INFO("lambda=%f, epochs=%d, eta0=%f\n", lambda, epochs, eta0)

	//do the sgd
	calibrate();
	if (features->is_seekable())
		features->reset_stream();

	ELossType loss_type = loss->get_loss_type();
	bool is_log_loss = false;
	if ((loss_type == L_LOGLOSS) || (loss_type == L_LOGLOSSMARGIN))
		is_log_loss = true;

	int32_t vec_count;
	for (auto e:progress(range(epochs)))
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
	SG_INFO("Norm: %.6f, Bias: %.6f\n", wnorm, bias)

	return true;
}

void COnlineSVMSGD::calibrate(int32_t max_vec_num)
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
		m=CMath::max(c, c_dim);
		n++;

		features->release_example();
		if (n>=max_vec_num || m > 1000)
			break;
	}

	SG_PRINT("Online SGD calibrated using %d vectors.\n", n)

	// bias update scaling
	bscale = 0.5*m/n;

	// compute weight decay skip
	skip = (int32_t) ((16 * n * c_dim) / r);

	SG_INFO("using %d examples. skip=%d  bscale=%.6f\n", n, skip, bscale)

	SG_FREE(c);
}

void COnlineSVMSGD::init()
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

	loss=new CHingeLoss();
	SG_REF(loss);

	SG_ADD(&C1, "C1", "Cost constant 1.", MS_AVAILABLE);
	SG_ADD(&C2, "C2", "Cost constant 2.", MS_AVAILABLE);
	SG_ADD(&lambda, "lambda", "Regularization parameter.", MS_AVAILABLE);
	SG_ADD(&wscale, "wscale", "W scale", MS_NOT_AVAILABLE);
	SG_ADD(&bscale, "bscale", "b scale", MS_NOT_AVAILABLE);
	SG_ADD(&epochs, "epochs", "epochs", MS_NOT_AVAILABLE);
	SG_ADD(&skip, "skip", "skip", MS_NOT_AVAILABLE);
	SG_ADD(&count, "count", "count", MS_NOT_AVAILABLE);
	SG_ADD(
	    &use_bias, "use_bias", "Indicates if bias is used.", MS_NOT_AVAILABLE);
	SG_ADD(
	    &use_regularized_bias, "use_regularized_bias",
	    "Indicates if bias is regularized.", MS_NOT_AVAILABLE);
}
