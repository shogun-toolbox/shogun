/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __DEEPAUTOENCODER_H__
#define __DEEPAUTOENCODER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/Autoencoder.h>

namespace shogun
{
template <class T> class DenseFeatures;

/** @brief Represents a muti-layer autoencoder
 *
 * A deep autoencoder consists of an input layer, multiple encoding layers, and
 * multiple decoding layers. It can be pre-trained as a stack of single layer
 * autoencoders. Fine-tuning can performed on the entire autoencoder in an
 * unsupervised manner using train(), or in a supervised manner using
 * convert_to_neural_network().
 *
 * This class supports training normal deep autoencoders and
 * [denoising autoencoders](http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/217)
 * [Vincent, 2008]. To use denoising autoencoders set noise_type and noise_parameter
 * to specify the type and strength of the noise (pt_noise_type and
 * pt_noise_parameter for pre-training).
 *
 * NOTE: LBFGS does not work properly with denoising autoencoders due to their
 * stochastic nature. Use gradient descent instead.
 *
 * [Deep contractive autoencoders](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf)
 * [Rifai, 2011] are also supported. To use them, call set_contraction_coefficient()
 * (or use pt_contraction_coefficient for pre-training).
 * Denoising can also be used with contractive autoencoders through noise_type
 * and noise_parameter.
 *
 * [Deep convolutional autoencoders](http://www.idsia.ch/~ciresan/data/icann2011.pdf)
 * [J Masci, 2011] are also supported. Simply build the autoencoder
 * using NeuralConvolutionalLayer objects.
 *
 * NOTE: Contractive convolutional autoencoders are not supported.
 *
 * If the autoencoder has N layers, encoding layers will be the layers following
 * the input layer up to and including layer (N-1)/2. The rest of the layers are
 * called the decoding layers. Note that the number of encoding layers is the
 * same as the number of decoding layers.
 *
 * The layers of the autoencoder must be symmetric in the number of neurons
 * about the last encoding layer, that is, layer i must have the same number of
 * neurons as layer N-i-1. For example, a valid structure could be something
 * like: 500->250->100->250->500.
 *
 * When finetuning the autoencoder in a unsupervised manner, denoising and
 * contraction can also be used through set_contraction_coefficient() and
 * noise_type and noise_parameter. See Autoencoder for more details.
 */
class DeepAutoencoder : public Autoencoder
{
public:
	/** default constructor */
	DeepAutoencoder();

	/** Constructs and initializes an autoencoder
	 *
	 * @param layers An array of NeuralLayer objects specifying the layers of
	 * the autoencoder
	 * @param sigma Standard deviation of the gaussian used to initialize the
	 * weights
	 */
	DeepAutoencoder(
		const std::vector<std::shared_ptr<NeuralLayer>>& layers,
		float64_t sigma = 0.01);

	~DeepAutoencoder() override {}

	/** Trains the DeepAutoencoder by calling pre_train if specified,
	 * followed by a call to Autoencoder::train.
	 *
	 * @param data Training examples
	 *
	 * @return True if training succeeded, false otherwise
	 */
	bool train(std::shared_ptr<Features> data) override;

	/** Pre-trains the deep autoencoder as a stack of autoencoders
	 *
	 * If the deep autoencoder has N layers, it is treated as a stack of (N-1)/2
	 * single layer autoencoders. For all \f$ 1<i<(N-1)/2 \f$ an autoencoder is
	 * formed using layer i-1 as an input layer, layer i as encoding layer,
	 * and layer N-i as decoding layer.
	 *
	 * For example, if the deep autoencoder has layers L0->L1->L2->L3->L4, two
	 * autoencoders will be formed: L0->L1->L4 and L1->L2->L3.
	 *
	 * Training parameters for each autoencoder can be set using the pt_*
	 * public fields, i.e pt_optimization_method and pt_contraction_coefficient.
	 * Each of those fields is a vector of length (N-1)/2, where the first
	 * element sets the parameter for the first autoencoder, the second element
	 * set the parameter for the second autoencoder and so on. When required,
	 * the parameter can be set for all autoencoders using the
	 * SGVector::set_const() method.
	 *
	 * @param data Training examples
	 */
	virtual void pre_train(std::shared_ptr<Features> data);

	/** Forward propagates the data through the autoencoder and returns the
	 * activations of the last encoding layer (layer (N-1)/2)
	 *
	 * @param data Input features
	 *
	 * @return Transformed features
	 */
	std::shared_ptr<DenseFeatures<float64_t>> transform(
		std::shared_ptr<DenseFeatures<float64_t>> data) override;

	/** Forward propagates the data through the autoencoder and returns the
	 * activations of the last layer
	 *
	 * @param data Input features
	 *
	 * @return Reconstructed features
	 */
	std::shared_ptr<DenseFeatures<float64_t>> reconstruct(
		std::shared_ptr<DenseFeatures<float64_t>> data) override;

	/** Helper function to initialize DeepAutoencoder
	 */
	void initialize_neural_network(float64_t sigma = 0.01f) override;

	/** Converts the autoencoder into a neural network for supervised finetuning.
	 *
	 * The neural network is formed using the input layer and the encoding layers.
	 * If specified, another output layer will added on top of those layers
	 *
	 * @param output_layer If specified, this layer will be added on top of the
	 * last encoding layer
	 * @param sigma Standard deviation used to initialize the parameters of the
	 * output layer
	 */
	virtual std::shared_ptr<NeuralNetwork> convert_to_neural_network(
		std::shared_ptr<NeuralLayer> output_layer=NULL, float64_t sigma = 0.01);

	/** Sets the contraction coefficient
	 *
	 * For contractive autoencoders [Rifai, 2011], a term:
	 * \f[ \frac{\lambda}{N} \sum_{k=0}^{N-1} \left \| J(x_k) \right \|^2_F \f]
	 * is added to the error, where \f$ \left \| J(x_k)) \right \|^2_F \f$ is the
	 * Frobenius norm of the Jacobian of the activations of the each encoding layer
	 * with respect to its inputs, \f$ N \f$ is the batch size, and
	 * \f$ \lambda \f$ is the contraction coefficient.
	 *
	 * @param coeff Contraction coefficient
	 */
	void set_contraction_coefficient(float64_t coeff) override;

	const char* get_name() const override { return "DeepAutoencoder"; }

protected:
	/** Computes the error between the output layer's activations and the given
	 * target activations.
	 *
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 */
	float64_t compute_error(SGMatrix<float64_t> targets) override;

private:
	void init();

	/** Returns the section of vector v that belongs to layer i */
	template<class T>
	SGVector<T> get_section(SGVector<T> v, int32_t i);

public:
	/** Autoencoder::noise_type for pre-training each encoding layer
	 * Default value is AENT_NONE for all layers
	 */
	SGVector<int32_t> pt_noise_type;

	/** Autoencoder::noise_parameter for pre-training each encoding layer
	 * Default value is 0.0 for all layers
	 */
	SGVector<float64_t> pt_noise_parameter;

	/** Contraction coefficient (see Autoencoder::set_contraction_coefficient())
	 * for pre-training each encoding layer
	 * Default value is 0.0 for all layers
	 */
	SGVector<float64_t> pt_contraction_coefficient;

	/** Autoencoder::optimization_method for pre-training each encoding layer
	 * Default value is NNOM_LBFGS for all layers
	 */
	SGVector<int32_t> pt_optimization_method;

	/** Autoencoder::l2_coefficient for pre-training each encoding layer
	 * Default value is 0.0 for all layers
	 */
	SGVector<float64_t> pt_l2_coefficient;

	/** Autoencoder::l1_coefficient for pre-training each encoding layer
	 * Default value is 0.0 for all layers
	 */
	SGVector<float64_t> pt_l1_coefficient;

	/** Autoencoder::epsilon for pre-training each encoding layer
	 * Default value is 1.0e-5 for all layers
	 */
	SGVector<float64_t> pt_epsilon;

	/** Autoencoder::max_num_epochs for pre-training each encoding layer
	 * Default value is 0 for all layers
	 */
	SGVector<int32_t> pt_max_num_epochs;

	/** Autoencoder::gd_mini_batch_size for pre-training each encoding layer
	 * Default value is 0 for all layers
	 */
	SGVector<int32_t> pt_gd_mini_batch_size;

	/** Autoencoder::gd_learning_rate for pre-training each encoding layer
	 * Default value is 0.1 for all layers
	 */
	SGVector<float64_t> pt_gd_learning_rate;

	/** Autoencoder::gd_learning_rate_decay for pre-training each encoding layer
	 * Default value is 1.0 for all layers
	 */
	SGVector<float64_t> pt_gd_learning_rate_decay;

	/** Autoencoder::gd_momentum for pre-training each encoding layer
	 * Default value is 0.9 for all layers
	 */
	SGVector<float64_t> pt_gd_momentum;

	/** Autoencoder::gd_error_damping_coeff for pre-training each encoding layer
	 * Default value is -1 for all layers
	 */
	SGVector<float64_t> pt_gd_error_damping_coeff;

	/** Whether to pretrain with relevant parameters */
	bool m_do_pretrain;

protected:
	/** Standard deviation of the gaussian used to initialize the
	 * parameters */
	float64_t m_sigma;
};
}
#endif
