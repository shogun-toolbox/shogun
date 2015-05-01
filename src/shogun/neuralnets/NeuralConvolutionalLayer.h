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

#ifndef __NEURALCONVOLUTIONALLAYER_H__
#define __NEURALCONVOLUTIONALLAYER_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/neuralnets/ConvolutionalFeatureMap.h>

namespace shogun
{
/** @brief Main component in [convolutional neural networks]
 * (http://en.wikipedia.org/wiki/Convolutional_neural_network)
 *
 * This layer type of consists of multiple feature maps. Each feature map
 * computes its activations using by convolving its filter with the inputs,
 * adding a bias, and then applying a non-linearity. Activations of each
 * feature map can be max-pooled, that is, the map is divided into regions of
 * a certain size and then the maximum activation is taken from each region.
 *
 * All layer that are connected to this layer as input must have the same size.
 *
 * During convolution, the inputs are implicitly padded with zeros on the
 * sides
 *
 * The layer assumes that its input images are in column major format
 */
class CNeuralConvolutionalLayer : public CNeuralLayer
{
public:
	/** default constructor */
	CNeuralConvolutionalLayer();

	/** Constuctor
	 *
	 * @param function Activation function
	 *
	 * @param num_maps Number of feature maps
	 *
	 * @param radius_x Radius of the convolution filter on the x (width) axis.
	 * The filter size on the x-axis equals (2*radius_x+1)
	 *
	 * @param radius_y Radius of the convolution filter on the y (height) axis.
	 * The filter size on the y-axis equals (2*radius_y+1)
	 *
	 * @param pooling_width Width of the pooling region
	 *
	 * @param pooling_height Height of the pooling region
	 *
	 * @param stride_x Stride in the x direction for convolution
	 *
	 * @param stride_y Stride in the y direction for convolution
	 */
	CNeuralConvolutionalLayer(EConvMapActivationFunction function,
		int32_t num_maps,
		int32_t radius_x, int32_t radius_y,
		int32_t pooling_width=1, int32_t pooling_height=1,
		int32_t stride_x=1, int32_t stride_y=1);

	virtual ~CNeuralConvolutionalLayer() {}

	/** Sets the batch_size and allocates memory for m_activations and
	 * m_input_gradients accordingly. Must be called before forward or backward
	 * propagation is performed
	 *
	 * @param batch_size number of training/test cases the network is
	 * currently working with
	 */
	virtual void set_batch_size(int32_t batch_size);

	/** Initializes the layer, computes the number of parameters needed for
	 * the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 * @param input_indices  Indices of the layers that are connected to this
	 * layer as input
	 */
	virtual void initialize(CDynamicObjectArray* layers,
			SGVector<int32_t> input_indices);

	/** Initializes the layer's parameters. The layer should fill the given
	 * arrays with the initial value for its parameters
	 *
	 * @param parameters Vector of size get_num_parameters()
	 *
	 * @param parameter_regularizable Vector of size get_num_parameters().
	 * This controls which of the layer's parameter are
	 * subject to regularization, i.e to turn off regularization for parameter
	 * i, set parameter_regularizable[i] = false. This is usally used to turn
	 * off regularization for bias parameters.
	 *
	 * @param sigma standard deviation of the gaussian used to random the
	 * parameters
	 */
	virtual void initialize_parameters(SGVector<float64_t> parameters,
			SGVector<bool> parameter_regularizable,
			float64_t sigma);

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 */
	virtual void compute_activations(SGVector<float64_t> parameters,
			CDynamicObjectArray* layers);

	/** Computes the gradients that are relevent to this layer:
	 *- The gradients of the error with respect to the layer's parameters
	 * -The gradients of the error with respect to the layer's inputs
	 *
	 * Input gradients for layer i that connects into this layer as input are
	 * added to m_layers.element(i).get_activation_gradients()
	 *
	 * Deriving classes should make sure to account for
	 * [dropout](http://arxiv.org/abs/1207.0580) [Hinton, 2012] during gradient
	 * computations
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param targets a matrix of size num_neurons*batch_size. If the layer is
	 * being used as an output layer, targets is the desired values for the
	 * layer's activations, otherwise it's an empty matrix
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 * @param parameter_gradients Vector of size get_num_parameters(). To be
	 * filled with gradients of the error with respect to each parameter of the
	 * layer
	 */
	virtual void compute_gradients(SGVector<float64_t> parameters,
			SGMatrix<float64_t> targets,
			CDynamicObjectArray* layers,
			SGVector<float64_t> parameter_gradients);

	/** Computes the error between the layer's current activations and the given
	 * target activations. Should only be used with output layers
	 *
	 * @param targets desired values for the layer's activations, matrix of size
	 * num_neurons*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets);

	/** Constrains the weights of each neuron in the layer to have an L2 norm of
	 * at most max_norm
	 *
	 * @param parameters pointer to the layer's parameters, array of size
	 * get_num_parameters()
	 *
	 * @param max_norm maximum allowable norm for a neuron's weights
	 */
	virtual void enforce_max_norm(SGVector<float64_t> parameters,
			float64_t max_norm);

	virtual const char* get_name() const { return "NeuralConvolutionalLayer"; }

private:
	void init();

protected:
	/** Number of feature maps */
	int32_t m_num_maps;

	/** Width of the input */
	int32_t m_input_width;

	/** Height of the input */
	int32_t m_input_height;

	/** Total number channels in the inputs */
	int32_t m_input_num_channels;

	/** Radius of the convolution filter on the x (width) axis */
	int32_t m_radius_x;

	/** Radius of the convolution filter on the y (height) axis */
	int32_t m_radius_y;

	/** Width of the pooling region */
	int32_t m_pooling_width;

	/** Height of the pooling region */
	int32_t m_pooling_height;

	/** Stride in the x direction */
	int32_t m_stride_x;

	/** Stride in the y direcetion */
	int32_t m_stride_y;

	/** The map's activation function */
	EConvMapActivationFunction m_activation_function;

	/** Holds the output of convolution */
	SGMatrix<float64_t> m_convolution_output;

	/** Gradients of the error with respect to the convolution's output */
	SGMatrix<float64_t> m_convolution_output_gradients;

	/** Row indices of the max elements for each pooling region */
	SGMatrix<float64_t> m_max_indices;
};

}
#endif
