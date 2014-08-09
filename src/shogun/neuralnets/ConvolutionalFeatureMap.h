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

#ifndef __CONVOLUTIONALFEATUREMAP_H__
#define __CONVOLUTIONALFEATUREMAP_H__

#include <shogun/lib/common.h>
#include <shogun/neuralnets/NeuralLayer.h>

namespace shogun
{

/** @brief Determines the activation function for neurons in a convolutional 
 * feature map
 */
enum EConvMapActivationFunction
{
	/** Identity activation function: \f$ f(x) = x \f$ */
	CMAF_IDENTITY = 0,
	
	/** Logistic activation function: \f$ f(x) = \frac{1}{1+exp(-x)} \f$ */ 
	CMAF_LOGISTIC = 1,
	
	/** Rectified linear activation function: \f$ f(x) = max(x,0) \f$ */
	CMAF_RECTIFIED_LINEAR = 2
};

template <class T> class SGVector;
template <class T> class SGMatrix;
class CDynamicObjectArray;

/** @brief Handles convolution and gradient calculation for a single feature 
 * map in a convolutional neural network
 */
class CConvolutionalFeatureMap
{
public:	
	/** Constuctor
	 * 
	 * @param input_width Width of the input
	 * 
	 * @param input_height Height of the input
	 * 
	 * @param radius_x Radius of the convolution filter on the x (width) axis
	 * 
	 * @param radius_y Radius of the convolution filter on the y (height) axis
	 * 
	 * @param stride_x Stride in the x direction
	 * 
	 * @param stride_y Stride in the y direction
	 * 
	 * @param index Index of this feature map in its layer. This affects which
	 * part of the activations/activation_gradients matrix the map will store 
	 * its outputs in.
	 * 
	 * @param function Activation function
	 */
	CConvolutionalFeatureMap(int32_t input_width, int32_t input_height, 
			int32_t radius_x, int32_t radius_y, 
			int32_t stride_x=1, int32_t stride_y=1,
			int32_t index=0,
			EConvMapActivationFunction function = CMAF_IDENTITY,
			ENLAutoencoderPosition autoencoder_position = NLAP_NONE);
	
	/** Computes the activations of the feature map
	 * 
	 * @param parameters Vector of parameters for the map. length 
	 * width*height+(2*radius_x+1)+(2*radius_y+1)
	 * 
	 * @param layers The layers array that forms the network in which the map
	 * is being used
	 * 
	 * @param input_indices Indices of the layers that are connected to the map
	 * as input
	 * 
	 * @param activations Matrix in which the activations are to be stored
	 */
	void compute_activations(SGVector<float64_t> parameters, 
			CDynamicObjectArray* layers,
			SGVector<int32_t> input_indices,
			SGMatrix<float64_t> activations);
	
	/** Computes the gradients with respect to the parameters and the inputs to 
	 * the map
	 * 
	 * @param parameters Vector of parameters for the map. length 
	 * width*height+(2*radius_x+1)+(2*radius_y+1)
	 * 
	 * @param activations Activations of the map
	 * 
	 * @param activation_gradients Gradients of the error with respect to the 
	 * map's activations
	 * 
	 * @param layers The layers array that forms the network in which the map
	 * is being used
	 * 
	 * @param input_indices Indices of the layers that are connected to the map
	 * as input
	 * 
	 * @param parameter_gradients Vector in which the parameters gradients are to be 
	 * stored
	 */
	void compute_gradients(SGVector<float64_t> parameters,
			SGMatrix<float64_t> activations,
			SGMatrix<float64_t> activation_gradients,
			CDynamicObjectArray* layers,
			SGVector<int32_t> input_indices,
			SGVector<float64_t> parameter_gradients);
	
	/** Applies max pooling to the activations
	 * 
	 * @param activations Activations of the map
	 * 
	 * @param pooling_width Width of the pooling region
	 * 
	 * @param pooling_height Height of the pooling region
	 * 
	 * @param pooled_activations Result of the pooling process
	 * 
	 * @param max_indices Row indices of the max elements for each pooling 
	 * region
	 */
	void pool_activations(SGMatrix<float64_t> activations,
			int32_t pooling_width, 
			int32_t pooling_height,
			SGMatrix<float64_t> pooled_activations,
			SGMatrix<float64_t> max_indices);
	
protected:
	/** Perfoms convolution
	 * 
	 * @param inputs Inputs matrix. Each column in the matrix is treated as an 
	 * image in column major format
	 * 
	 * @param weights Convolution filter
	 * 
	 * @param outputs Output matrix 
	 * 
	 * @param flip If true the weights are flipped, performing cross-correlation 
	 * instead of convolution
	 * 
	 * @param reset_output true the output is reset to zero before performing 
	 * convolution
	 * 
	 * @param inputs_row_offset Index of the row at which the input image starts
	 * 
	 * @param outputs_row_offset Index of the row at which the output image starts
	 */
	void convolve(SGMatrix<float64_t> inputs, 
			SGMatrix<float64_t> weights, 
			SGMatrix<float64_t> outputs,
			bool flip,
			bool reset_output,
			int32_t inputs_row_offset,
 			int32_t outputs_row_offset);
	
	/** Computes the gradients of the error with respect to the weights, for a 
	 * particular input matrix
	 * 
	 * @param inputs Inputs matrix
	 * 
	 * @param local_gradients Gradients with respect the map's pre-activations
	 * 
	 * @param weight_gradients Matrix to store the gradients in
	 * 
	 * @param inputs_row_offset Offset for accessing the rows of the inputs 
	 * matrix
	 * 
	 * @param local_gradients_row_offset Offset for accessing the rows of the 
	 * local gradients matrix
	 */
	void compute_weight_gradients(SGMatrix<float64_t> inputs, 
			SGMatrix<float64_t> local_gradients, 
			SGMatrix<float64_t> weight_gradients,
			int32_t inputs_row_offset,
 			int32_t local_gradients_row_offset);
	
protected:
	/** Width of the input */
	int32_t m_input_width;
	
	/** Height of the input */
	int32_t m_input_height;
	
	/** Radius of the convolution filter on the x (width) axis */
	int32_t m_radius_x;
	
	/** Radius of the convolution filter on the y (height) axis */
	int32_t m_radius_y;
	
	/** Stride in the x direction */
	int32_t m_stride_x;
	
	/** Stride in the y direcetion */
	int32_t m_stride_y;
	
	/** Index of this feature map in its layer. This affects which
	 * part of the activations/activation_gradients matrix that map will use
	 */
	int32_t m_index;
	
	/** The map's activation function */
	EConvMapActivationFunction m_activation_function;
	
	/** Width of the convolution's output image */
	int32_t m_output_width;
	
	/** Height of the convolution's output image */
	int32_t m_output_height;
	
	/** Number of neurons in the input */
	int32_t m_input_num_neurons;
	
	/** Number of neurons in the output */
	int32_t m_output_num_neurons;
	
	/** Row offset for accessing the activations */
	int32_t m_row_offset;
	
	/** Width of the convolution filter */
	int32_t m_filter_width;
	
	/** Height of the convolution filter */
	int32_t m_filter_height;
	
	/** For autoencoders, specifies the position of the layer in the autoencoder, 
	 * i.e an encoding layer or a decoding layer. Default value is NLAP_NONE
	 */
	ENLAutoencoderPosition m_autoencoder_position;
};

}
#endif
