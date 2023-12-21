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
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{
template<class T> class DenseFeatures;
class NeuralLayer;

/** optimization method for neural networks */
enum ENNOptimizationMethod
{
	NNOM_GRADIENT_DESCENT=0,
	NNOM_LBFGS=1
};

/** @brief A generic multi-layer neural network
 *
 * A [Neural network](http://en.wikipedia.org/wiki/Artificial_neural_network)
 * is constructed using an array of NeuralLayer objects. The NeuralLayer
 * class defines the interface necessary for forward and
 * [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
 *
 * The network can be constructed as any arbitrary directed acyclic graph.
 *
 * How to use the network:
 * 	- Prepare an array of NeuralLayer-based objects that specify
 * the type of layers used in the network. The array must contain at least one
 * input layer. The last layer in the array is treated as the output layer.
 * Also note that forward propagation is performed in the order at which the
 * layers appear in the array. So if layer j takes its input from layer i then
 * i must be less than j.
 * 	- Specify how the layers are connected together. This can be done using
 * either connect() or quick_connect().
 * 	- Call initialize_neural_network()
 * 	- Specify the training parameters if needed
 * 	- Train set_labels() and train()
 * 	- If needed, the network with the learned parameters can be stored on disk
 * using save_serializable() (loaded using load_serializable())
 * 	- Apply the network using apply()
 *
 * The network can also be initialized from a JSON file using
 * NeuralNetworkFileReader.
 *
 * Supported feature types: DenseFeatures<float64_t>
 * Supported label types:
 * 	- BinaryLabels
 * 	- MulticlassLabels
 * 	- RegressionLabels
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
 * on info messages call sg_io->set_loglevel(MSG_INFO)).
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
class NeuralNetwork : public RandomMixin<Machine>
{
friend class DeepBeliefNetwork;

public:
	/** default constuctor */
	NeuralNetwork();

	/** Sets the layers of the network
	 *
	 * @param layers An array of NeuralLayer objects specifying the layers of
	 * the network. Must contain at least one input layer. The last layer in
	 * the array is treated as the output layer
	 */
	NeuralNetwork(const std::vector<std::shared_ptr<NeuralLayer>>& layers);

	/** Sets the layers of the network
	 *
	 * @param layers An array of NeuralLayer objects specifying the layers of
	 * the network. Must contain at least one input layer. The last layer in
	 * the array is treated as the output layer
	 */
	virtual void
	set_layers(const std::vector<std::shared_ptr<NeuralLayer>>& layers);

	/** Connects layer i as input to layer j. In order for forward and
	 * backpropagation to work correctly, i must be less that j
	 */
	virtual void connect(int32_t i, int32_t j);

	/** Initialize adjacency matrix
	 */
	virtual void init_adj_matrix();

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
	virtual void initialize_neural_network(float64_t sigma = 0.01f);

	~NeuralNetwork() override;

	/** apply machine to data in means of binary classification problem */
	std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data) override;
	/** apply machine to data in means of regression problem */
	std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data) override;
	/** apply machine to data in means of multiclass classification problem */
	std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data) override;

	/** Applies the network as a feature transformation
	 *
	 * Forward-propagates the data through the network and returns the
	 * activations of the last layer
	 *
	 * @param data Input features
	 *
	 * @return Transformed features
	 */
	virtual std::shared_ptr<DenseFeatures<float64_t>> transform(
		std::shared_ptr<DenseFeatures<float64_t>> data);

	/** get classifier type
	 *
	 * @return classifier type CT_NEURALNETWORK
	 */
	EMachineType get_classifier_type() override { return CT_NEURALNETWORK; }

	/** returns type of problem machine solves */
	EProblemType get_machine_problem_type() const override;

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
	SGVector<float64_t>* get_layer_parameters(int32_t i) const;

	/** returns a copy of all the layer's parameters array
	 *
	 * @param i index of the layer
	 */
	std::vector<SGVector<float64_t>> get_layer_parameters() const;

	/** returns the totat number of parameters in the network */
	int32_t get_num_parameters() const { return m_total_num_parameters; }

	/** return the network's parameter array */
	SGVector<float64_t> get_parameters() const { return m_params; }

	/** returns the number of inputs the network takes*/
	int32_t get_num_inputs() const { return m_num_inputs; }

	/** returns the number of neurons in the output layer */
	int32_t get_num_outputs() const;

	/** Returns an array holding the network's layers */
	const std::vector<std::shared_ptr<NeuralLayer>>& get_layers() const;

	const char* get_name() const override { return "NeuralNetwork";}

	/** Sets optimization method
	 * default is NNOM_LBFGS
	 * @param optimization_method optimiation method
	 */
	void set_optimization_method(ENNOptimizationMethod optimization_method)
	{
		m_optimization_method = optimization_method;
	}

	/** Returns optimization method */
	ENNOptimizationMethod get_optimization_method() const
	{
		return m_optimization_method;
	}
	/** Sets L2 Regularization coeff
	 * default value is 0.0
	 * @param l2_coefficient l2_coefficient
	 */
	void set_l2_coefficient(float64_t l2_coefficient)
	{
		m_l2_coefficient = l2_coefficient;
	}

	/** Returns L2 coefficient */
	float64_t get_l2_coefficient() const
	{
		return m_l2_coefficient;
	}
	/** Sets L1 Regularization coeff
	 * default value is 0.0
	 * @param l1_coefficient l1_coefficient
	 */
	void set_l1_coefficient(float64_t l1_coefficient)
	{
		m_l1_coefficient = l1_coefficient;
	}

	/** Returns L1 coefficient */
	float64_t get_l1_coefficient() const
	{
		return m_l1_coefficient;
	}

	/** Sets the probabilty that a hidden layer neuron will be dropped out
	 * When using this, the recommended value is 0.5
	 * default value 0.0 (no dropout)
	 *
	 * For more details on dropout, see
	 * [paper](http://arxiv.org/abs/1207.0580) [Hinton, 2012]
	 *
	 * @param dropout_hidden dropout probability
	 */
	void set_dropout_hidden(float64_t dropout_hidden)
	{
		m_dropout_hidden = dropout_hidden;
	}

	/** Returns dropout probability for hidden layers */
	float64_t get_dropout_hidden() const
	{
		return m_dropout_hidden;
	}

	/** Sets the probabilty that an input layer neuron will be dropped out
	 * When using this, a good value might be 0.2
	 * default value 0.0 (no dropout)
	 *
	 * For more details on dropout, see this
	 * [paper](http://arxiv.org/abs/1207.0580) [Hinton, 2012]
	 *
	 * @param dropout_input dropout probability
	 */
	void set_dropout_input(float64_t dropout_input)
	{
		m_dropout_input = dropout_input;
	}

	/** Returns dropout probability for input layers */
	float64_t get_dropout_input() const
	{
		return m_dropout_input;
	}

	/** Sets maximum allowable L2 norm for a neurons weights
	 * When using this, a good value might be 15
	 * default value -1 (max-norm regularization disabled)
	 * @param max_norm maximum allowable L2 norm
	 */
	void set_max_norm(float64_t max_norm)
	{
		m_max_norm = max_norm;
	}

	/** Returns maximum allowable L2 norm */
	float64_t get_max_norm() const
	{
		return m_max_norm;
	}

	/** Sets convergence criteria
	 * training stops when (E'- E)/E < epsilon
	 * where E is the error at the current iterations and E' is the error at the
	 * previous iteration
	 * default value is 1.0e-5
	 * @param epsilon convergence criteria
	 */
	void set_epsilon(float64_t epsilon)
	{
		m_epsilon = epsilon;
	}

	/** Returns epsilon */
	float64_t get_epsilon() const
	{
		return m_epsilon;
	}

	/** Sets maximum number of iterations over the training set.
	 * If 0, training will continue until convergence.
	 * defualt value is 0
	 * @param max_num_epochs maximum number of iterations over the training set
	 */
	void set_max_num_epochs(int32_t max_num_epochs)
	{
		m_max_num_epochs = max_num_epochs;
	}

	/** Returns maximum number of epochs */
	int32_t get_max_num_epochs() const
	{
		return m_max_num_epochs;
	}

	/** Sets size of the mini-batch used during gradient descent training,
	 * if 0 full-batch training is performed
	 * default value is 0
	 * @param gd_mini_batch_size mini batch size
	 */
	void set_gd_mini_batch_size(int32_t gd_mini_batch_size)
	{
		m_gd_mini_batch_size = gd_mini_batch_size;
	}

	/** Returns mini batch size */
	int32_t get_gd_mini_batch_size() const
	{
		return m_gd_mini_batch_size;
	}

	/** Sets gradient descent learning rate
	 * defualt value 0.1
	 * @param gd_learning_rate gradient descent learning rate
	 */
	void set_gd_learning_rate(float64_t gd_learning_rate)
	{
		m_gd_learning_rate = gd_learning_rate;
	}

	/** Returns gradient descent learning rate */
	float64_t get_gd_learning_rate() const
	{
		return m_gd_learning_rate;
	}

	/** Sets gradient descent learning rate decay
	 * learning rate is updated at each iteration i according to:
	 * alpha(i)=decay*alpha(i-1)
	 * default value is 1.0 (no decay)
	 * @param gd_learning_rate_decay gradient descent learning rate decay
	 */
	void set_gd_learning_rate_decay(float64_t gd_learning_rate_decay)
	{
		m_gd_learning_rate_decay = gd_learning_rate_decay;
	}

	/** Returns gradient descent learning rate decay */
	float64_t get_gd_learning_rate_decay() const
	{
		return m_gd_learning_rate_decay;
	}

	/** Sets gradient descent momentum multiplier
	 *
	 * default value is 0.9
	 *
	 * For more details on momentum, see this
	 * [paper](http://jmlr.org/proceedings/papers/v28/sutskever13.html)
	 * [Sutskever, 2013]
	 *
	 * @param gd_momentum gradient descent momentum multiplier
	 */
	void set_gd_momentum(float64_t gd_momentum)
	{
		m_gd_momentum = gd_momentum;
	}

	/** Returns gradient descent momentum multiplier */
	float64_t get_gd_momentum() const
	{
		return m_gd_momentum;
	}

	/** Sets gradient descent error damping coefficient
	 * Used to damp the error fluctuations when stochastic gradient descent is
	 * used. damping is done according to:
	 * error_damped(i) = c*error(i) + (1-c)*error_damped(i-1)
	 * where c is the damping coefficient
	 *
	 * If -1, the damping coefficient is automatically computed according to:
	 * c = 0.99*gd_mini_batch_size/training_set_size + 1e-2;
	 *
	 * default value is -1
	 *
	 * @param gd_error_damping_coeff error damping coefficient
	 */
	void set_gd_error_damping_coeff(float64_t gd_error_damping_coeff)
	{
		m_gd_error_damping_coeff = gd_error_damping_coeff;
	}

	float64_t get_gd_error_damping_coeff() const
	{
		return m_gd_error_damping_coeff;
	}

protected:
	/** trains the network */
	bool train_machine(const std::shared_ptr<Features>& data, const std::shared_ptr<Labels>& labs) override;

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
	virtual SGMatrix<float64_t> forward_propagate(std::shared_ptr<Features> data, int32_t j=-1);

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

	bool is_label_valid(std::shared_ptr<Labels >lab) const override;

	/** returns a pointer to layer i in the network */
	std::shared_ptr<NeuralLayer> get_layer(int32_t i) const;

	/** Ensures the given features are suitable for use with the network and
	 * returns their feature matrix
	 */
	SGMatrix<float64_t> features_to_matrix(const std::shared_ptr<Features>& features);

	/** converts the given labels into a matrix suitable for use with network
	 *
	 * @return matrix of size get_num_outputs()*num_labels
	 */
	SGMatrix<float64_t> labels_to_matrix(const std::shared_ptr<Labels>& labs);

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
	SGVector<T> get_section(SGVector<T> v, int32_t i) const;

protected:
	/** number of neurons in the input layer */
	int32_t m_num_inputs;

	/** number of layer */
	int32_t m_num_layers;

	/** network's layers */
	std::vector<std::shared_ptr<NeuralLayer>> m_layers;

	/** Describes the connections in the network: if there's a connection from
	 * layer i to layer j then m_adj_matrix(i,j) = 1.
	 */
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

	/** True if the network layers are to be quick connected and initialized
	 * initial value is False
	 */
	bool m_auto_quick_initialize;

	/** Standard deviation of the gaussian used to randomly
	* initialize the parameters
	*/
	float64_t m_sigma;

	/** Optimization method, default is NNOM_LBFGS */
	ENNOptimizationMethod m_optimization_method;

	/** L2 Regularization coeff, default value is 0.0*/
	float64_t m_l2_coefficient;

	/** L1 Regularization coeff, default value is 0.0*/
	float64_t m_l1_coefficient;

	/** Probabilty that a hidden layer neuron will be dropped out
	 * When using this, the recommended value is 0.5
	 *
	 * default value 0.0 (no dropout)
	 *
	 * For more details on dropout, see
	 * [paper](http://arxiv.org/abs/1207.0580) [Hinton, 2012]
	 */
	float64_t m_dropout_hidden;

	/** Probabilty that a input layer neuron will be dropped out
	 * When using this, a good value might be 0.2
	 *
	 * default value 0.0 (no dropout)
	 *
	 * For more details on dropout, see this
	 * [paper](http://arxiv.org/abs/1207.0580) [Hinton, 2012]
	 */
	float64_t m_dropout_input;

	/** Maximum allowable L2 norm for a neurons weights
	 *When using this, a good value might be 15
	 *
	 * default value -1 (max-norm regularization disabled)
	 */
	float64_t m_max_norm;

	/** convergence criteria
	 * training stops when (E'- E)/E < epsilon
	 * where E is the error at the current iterations and E' is the error at the
	 * previous iteration
	 * default value is 1.0e-5
	 */
	float64_t m_epsilon;

	/** maximum number of iterations over the training set.
	 * If 0, training will continue until convergence.
	 * defualt value is 0
	 */
	int32_t m_max_num_epochs;

	/** size of the mini-batch used during gradient descent training,
	 * if 0 full-batch training is performed
	 * default value is 0
	 */
	int32_t m_gd_mini_batch_size;

	/** gradient descent learning rate, defualt value 0.1 */
	float64_t m_gd_learning_rate;

	/** gradient descent learning rate decay
	 * learning rate is updated at each iteration i according to:
	 * alpha(i)=decay*alpha(i-1)
	 * default value is 1.0 (no decay)
	 */
	float64_t m_gd_learning_rate_decay;

	/** gradient descent momentum multiplier
	 *
	 * default value is 0.9
	 *
	 * For more details on momentum, see this
	 * [paper](http://jmlr.org/proceedings/papers/v28/sutskever13.html)
	 * [Sutskever, 2013]
	 */
	float64_t m_gd_momentum;

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
	float64_t m_gd_error_damping_coeff;

private:
	/** temperary pointers to the training data, used to pass the data to L-BFGS
	 * routines
	 */
	const SGMatrix<float64_t>* m_lbfgs_temp_inputs;
	const SGMatrix<float64_t>* m_lbfgs_temp_targets;

	EProblemType m_problem_type;
};

}
#endif
