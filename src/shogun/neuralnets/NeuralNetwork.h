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

#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
template<class T> class CDenseFeatures;
class CDynamicObjectArray;
class CNeuralLayer;

/** optimization method for neural networks */
enum ENNOptimizationMethod
{
	NNOM_GRADIENT_DESCENT=0,
	NNOM_LBFGS=1
};	

/** @brief A generic multi-layer neural network
 *
 * A [Neural network](http://en.wikipedia.org/wiki/Artificial_neural_network) 
 * is constructed using an array of CNeuralLayer objects. The NeuralLayer 
 * class defines the interface necessary for forward and 
 * [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
 * 
 * The network can be constructed as any arbitrary directed acyclic graph.  
 * 
 * How to use the network:
 * 	- Prepare a CDynamicObjectArray of CNeuralLayer-based objects that specify 
 * the type of layers used in the network. The array must contain at least one 
 * input layer. The last layer in the array is treated as the output layer. 
 * Also note that forward propagation is performed in the order at which the 
 * layers appear in the array. So if layer j takes its input from layer i then 
 * i must be less than j.
 * 	- Specify how the layers are connected together. This can be done using 
 * either connect() or quick_connect().
 * 	- Call initialize()
 * 	- Specify the training parameters if needed
 * 	- Train set_labels() and train()
 * 	- If needed, the network with the learned parameters can be stored on disk 
 * using save_serializable() (loaded using load_serializable())
 * 	- Apply the network using apply()
 * 
 * Supported feature types: CDenseFeatures<float64_t>
 * Supported label types:
 * 	- CBinaryLabels
 * 	- CMulticlassLabels
 * 	- CRegressionLabels
 * 
 * The neural network can be trained using 
 * [L-BFGS](http://en.wikipedia.org/wiki/Limited-memory_BFGS) (default) or 
 * [mini-batch gradient descent]
 * (http://en.wikipedia.org/wiki/Stochastic_gradient_descent).
 * 
 * NOTE: LBFGS does not work properly when using dropout/max-norm regularization
 * due to their stochastic nature. Use gradient descent instead.
 * 
 * During training, the error at each iteration is logged as MSG_INFO. (to turn
 * on info messages call io.set_loglevel(MSG_INFO)).
 * 
 * The network stores the parameters of all the  layers in a single array. This 
 * makes it easy to train a network of any combination of arbitrary layer types 
 * using any optimization method (gradient descent, L-BFGS, ..)
 * 
 * All the matrices the network (and related classes) deal with are in 
 * column-major format
 * 
 * When implemnting new layer types, the function check_gradients() can be used
 * to make sure the gradient computations are correct.
 */
class CNeuralNetwork : public CMachine
{
public:
	/** default constuctor */
	CNeuralNetwork();
	
	/** Sets the layers of the network
	 * 
	 * @param layers An array of CNeuralLayer objects specifying the layers of 
	 * the network. Must contain at least one input layer. The last layer in 
	 * the array is treated as the output layer
	 */
	CNeuralNetwork(CDynamicObjectArray* layers);
	
	/** Sets the layers of the network
	 * 
	 * @param layers An array of CNeuralLayer objects specifying the layers of 
	 * the network. Must contain at least one input layer. The last layer in 
	 * the array is treated as the output layer
	 */
	virtual void set_layers(CDynamicObjectArray* layers);
	
	/** Connects layer i as input to layer j. In order for forward and 
	 * backpropagation to work correctly, i must be less that j
	 */
	virtual void connect(int32_t i, int32_t j);
	
	/** Connects each layer to the layer after it. That is, connects layer i to 
	 * as input to layer i+1 for all i.
	 */
	virtual void quick_connect();
	
	/** Disconnects layer i from layer j */
	virtual void disconnect(int32_t i, int32_t j);
	
	/** Removes all connections in the network */
	virtual void disconnect_all();
	
	/** Initializes the network
	 * 
	 * @param sigma standard deviation of the gaussian used to randomly 
	 * initialize the parameters
	 */
	virtual void initialize(float64_t sigma = 0.01f);
	
	virtual ~CNeuralNetwork();
	
	/** apply machine to data in means of binary classification problem */
	virtual CBinaryLabels* apply_binary(CFeatures* data);
	/** apply machine to data in means of regression problem */
	virtual CRegressionLabels* apply_regression(CFeatures* data);
	/** apply machine to data in means of multiclass classification problem */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data);
	
	/** set labels
	*
	* @param lab labels
	*/
	virtual void set_labels(CLabels* lab);
	
	/** get classifier type
	 *
	 * @return classifier type CT_NEURALNETWORK
	 */
	virtual EMachineType get_classifier_type() { return CT_NEURALNETWORK; }
	
	/** returns type of problem machine solves */
	virtual EProblemType get_machine_problem_type() const;
	
	/** Checks if the gradients computed using backpropagation are correct by 
	 * comparing them with gradients computed using numerical approximation.
	 * Used for testing purposes only.
	 * 
	 * Gradients are numerically approximated according to:
	 * \f[ c = max(\epsilon x, s) \f]
	 * \f[ f'(x) = \frac{f(x + c)-f(x - c)}{2c} \f]
	 * 
	 * @param approx_epsilon Constant used during gradient approximation
	 * 
	 * @param s Some small value, used to prevent division by zero
	 * 
	 * @return Average difference between numerical gradients and 
	 * backpropagation gradients
	 */
	virtual float64_t check_gradients(float64_t approx_epsilon=1.0e-3, 
			float64_t s = 1.0e-9);
	
	/** returns a copy of a layer's parameters array 
	 * 
	 * @param i index of the layer
	 */
	SGVector<float64_t>* get_layer_parameters(int32_t i);
	
	/** returns the totat number of parameters in the network */
	int32_t get_num_parameters() { return m_total_num_parameters; }
	
	/** return the network's parameter array */
	SGVector<float64_t> get_parameters() { return m_params; }
	
	/** returns the number of inputs the network takes*/
	int32_t get_num_inputs() { return m_num_inputs; }
	
	/** returns the number of neurons in the output layer */
	int32_t get_num_outputs();
	
	CDynamicObjectArray* get_layers();
	
	virtual const char* get_name() const { return "NeuralNetwork";}
	
protected:	
	/** trains the network */
	virtual bool train_machine(CFeatures* data=NULL);
	
	/** trains the network using gradient descent*/
	virtual bool train_gradient_descent(SGMatrix<float64_t> inputs, 
			SGMatrix<float64_t> targets);
	
	/** trains the network using L-BFGS*/
	virtual bool train_lbfgs(SGMatrix<float64_t> inputs, 
			SGMatrix<float64_t> targets);
	
	/** Applies forward propagation, computes the activations of each layer up 
	 * to layer j
	 * 
	 * @param data input features
	 * @param j layer index at which the propagation should stop. If -1, the 
	 * propagation continues up to the last layer
	 * 
	 * @return activations of the last layer
	 */
	virtual SGMatrix<float64_t> forward_propagate(CFeatures* data, int32_t j=-1);
	
	/** Applies forward propagation, computes the activations of each layer up 
	 * to layer j
	 * 
	 * @param inputs inputs to the network, a matrix of size 
	 * m_num_inputs*m_batch_size
	 * @param j layer index at which the propagation should stop. If -1, the 
	 * propagation continues up to the last layer
	 * 
	 * @return activations of the last layer
	 */
	virtual SGMatrix<float64_t> forward_propagate(SGMatrix<float64_t> inputs, int32_t j=-1);
	
	/** Sets the batch size (the number of train/test cases) the network is 
	 * expected to deal with. 
	 * Allocates memory for the activations, local gradients, input gradients
	 * if necessary (if the batch size is different from it's previous value)
	 * 
	 * @param batch_size number of train/test cases the network is expected to 
	 * deal with.
	 */
	virtual void set_batch_size(int32_t batch_size);
	
	/** Applies backpropagation to compute the gradients of the error with
	 * repsect to every parameter in the network.
	 *
	 * @param inputs inputs to the network, a matrix of size 
	 * m_num_inputs*m_batch_size
	 * 
	 * @param targets desired values for the output layer's activations. matrix 
	 * of size m_layers[m_num_layers-1].get_num_neurons()*m_batch_size
	 * 
	 * @param gradients array to be filled with gradient values.
	 * 
	 * @return error between the targets and the activations of the last layer
	 */
	virtual float64_t compute_gradients(SGMatrix<float64_t> inputs, 
			SGMatrix<float64_t> targets, SGVector<float64_t> gradients);
	
	/** Forward propagates the inputs and computes the error between the output 
	 * layer's activations and the given target activations.
	 * 
	 * @param inputs inputs to the network, a matrix of size 
	 * m_num_inputs*m_batch_size
	 * 
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> inputs, 
			SGMatrix<float64_t> targets);
	
	/** Computes the error between the output layer's activations and the given
	 * target activations.
	 * 
	 * @param targets desired values for the network's output, matrix of size
	 * num_neurons_output_layer*batch_size
	 */
	virtual float64_t compute_error(SGMatrix<float64_t> targets);
	
	virtual bool is_label_valid(CLabels *lab) const;
	
	/** returns a pointer to layer i in the network */
	CNeuralLayer* get_layer(int32_t i);
	
	/** Ensures the given features are suitable for use with the network and 
	 * returns their feature matrix
	 */
	SGMatrix<float64_t> features_to_matrix(CFeatures* features);
	
	/** converts the given labels into a matrix suitable for use with network
	 * 
	 * @return matrix of size get_num_outputs()*num_labels
	 */
	SGMatrix<float64_t> labels_to_matrix(CLabels* labs);

private:
	void init();
	
	/** callback for l-bfgs */
	static float64_t lbfgs_evaluate(void *userdata, 
			const float64_t *W, 
			float64_t *grad, 
			const int32_t n, 
			const float64_t step);

	/** callback for l-bfgs */
	static int lbfgs_progress(void *instance,
			const float64_t *x,
			const float64_t *g,
			const float64_t fx,
			const float64_t xnorm,
			const float64_t gnorm,
			const float64_t step,
			int n,
			int k,
			int ls
			);
	
	/** Returns the section of vector v that belongs to layer i */
	template<class T>
	SGVector<T> get_section(SGVector<T> v, int32_t i);
public:
	/** Optimization method, default is NNOM_LBFGS */
	ENNOptimizationMethod optimization_method;
	
	/** L2 Regularization coeff, default value is 0.0*/
	float64_t l2_coefficient;
	
	/** L1 Regularization coeff, default value is 0.0*/
	float64_t l1_coefficient;
	
	/** Probabilty that a hidden layer neuron will be dropped out
	 * When using this, the recommended value is 0.5
	 * 
	 * default value 0.0 (no dropout)
	 * 
	 * For more details on dropout, see  
	 * [paper](http://arxiv.org/abs/1207.0580) [Hinton, 2012]
	 */
	float64_t dropout_hidden;
	
	/** Probabilty that a input layer neuron will be dropped out
	 * When using this, a good value might be 0.2
	 *
	 * default value 0.0 (no dropout)
	 * 
	 * For more details on dropout, see this 
	 * [paper](http://arxiv.org/abs/1207.0580) [Hinton, 2012]
	 */
	float64_t dropout_input;
	
	/** Maximum allowable L2 norm for a neurons weights
	 *When using this, a good value might be 15
	 * 
	 * default value -1 (max-norm regularization disabled)
	 */
	float64_t max_norm;
	
	/** convergence criteria
	 * training stops when (E'- E)/E < epsilon
	 * where E is the error at the current iterations and E' is the error at the
	 * previous iteration
	 * default value is 1.0e-5
	 */
	float64_t epsilon;
	
	/** maximum number of iterations over the training set.
	 * If 0, training will continue until convergence. 
	 * defualt value is 0
	 */
	int32_t max_num_epochs;
	
	/** size of the mini-batch used during gradient descent training, 
	 * if 0 full-batch training is performed
	 * default value is 0
	 */
	int32_t gd_mini_batch_size;
	
	/** gradient descent learning rate, defualt value 0.1 */
	float64_t gd_learning_rate;
	
	/** gradient descent learning rate decay
	 * learning rate is updated at each iteration i according to: 
	 * alpha(i)=decay*alpha(i-1)
	 * default value is 1.0 (no decay)
	 */
	float64_t gd_learning_rate_decay;
	
	/** gradient descent momentum multiplier
	 * 
	 * default value is 0.9
	 * 
	 * For more details on momentum, see this 
	 * [paper](http://jmlr.org/proceedings/papers/v28/sutskever13.html) 
	 * [Sutskever, 2013]
	 */
	float64_t gd_momentum;
	
	/** Used to damp the error fluctuations when stochastic gradient descent is 
	 * used. damping is done according to: 
	 * error_damped(i) = c*error(i) + (1-c)*error_damped(i-1)
	 * where c is the damping coefficient
	 * 
	 * If -1, the damping coefficient is automatically computed according to:
	 * c = 0.99*gd_mini_batch_size/training_set_size + 1e-2;
	 * 
	 * default value is -1
	 */
	float64_t gd_error_damping_coeff;
protected:
	/** number of neurons in the input layer */
	int32_t m_num_inputs;
	
	/** number of layer */
	int32_t m_num_layers;
	
	/** network's layers */
	CDynamicObjectArray* m_layers;
	
	SGMatrix<bool> m_adj_matrix;
	
	/** total number of parameters in the network */
	int32_t m_total_num_parameters;
	
	/** array where all the parameters of the network are stored */
	SGVector<float64_t> m_params;
	
	/** Array that specifies which parameters are to be regularized. This is 
	 * used to turn off regularization for bias parameters
	 */
	SGVector<bool> m_param_regularizable;
	
	/** offsets specifying where each layer's parameters and parameter 
	 * gradients are stored, i.e layer i's parameters are stored at 
	 * m_params + m_index_offsets[i]
	 */
	SGVector<int32_t> m_index_offsets;
	
	/** number of train/test cases the network is expected to deal with.
	 * Default value is 1
	 */
	int32_t m_batch_size;
	
	/** True if the network is currently being trained
	 * initial value is false
	 */
	bool m_is_training;
	
private:
	/** temperary pointers to the training data, used to pass the data to L-BFGS
	 * routines 
	 */
	const SGMatrix<float64_t>* m_lbfgs_temp_inputs;
	const SGMatrix<float64_t>* m_lbfgs_temp_targets;
};
	
}
#endif
