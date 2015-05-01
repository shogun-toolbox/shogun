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

#ifndef __DEEPBELIEFNETWORK_H__
#define __DEEPBELIEFNETWORK_H__

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/neuralnets/RBM.h>
#include <lib/SGMatrixList.h>

namespace shogun
{
template <class T> class SGVector;
template <class T> class SGMatrix;
template <class T> class SGMatrixList;
template <class T> class CDenseFeatures;
template <class T> class CDynamicArray;
class CDynamicObjectArray;
class CNeuralNetwork;
class CNeuralLayer;

/** @brief A Deep Belief Network
 *
 * A [Deep Belief Network](http://www.scholarpedia.org/article/Deep_belief_networks)
 * [Hinton, 2006] is a multilayer probabilistic generative models. It consists
 * of hidden layers and visible layers. The top hidden layer and the layer
 * below it form a Restricted Boltzmann Machine. The rest of
 * connections in the network are directed connections that go from a hidden
 * layer into a visible layer or another hidden layer.
 *
 * The network can be pre-trained by treating it as a stack of RBMs. Each hidden
 * layer along with the layer below it form an RBM. Each RBM
 * is then trained using (persistent) contrastive divergence. Pre-training often
 * provides a good initialization for the network's parameters.
 *
 * After pre-training, the parameters can be fine-tuned using a variant of the
 * wake-sleep algorithm [Hinton, 2006].
 *
 * The DBN can be used to initialize the parameters of a neural network using
 * convert_to_neural_network().
 *
 * Samples can be drawn from the model by starting with a random state for the
 * top hidden layer, performing some steps of Gibbs sampling in the top RBM to
 * obtain the states of the top hidden layer and then using those to infer the
 * states of the lower layers using a down-pass.
 *
 * Steps for using the DBN class:
 * 	- Specify the number of visible units and their type using the constructor
 * 	- Add the hidden layers using add_visible_layer() and add_hidden_layer().
 * 	- Initialize the DBN using initialize()
 * 	- Pre-train the DBN using pre_train()
 * 	- Optionally do some unsupervised fine-tuning using train()
 * 	- If needed, call convert_to_neural_network() to convert the DBN into a
 * neural network
 *
 */
class CDeepBeliefNetwork : public CSGObject
{
public:
	/** default constructor */
	CDeepBeliefNetwork();

	/** Creates a network with one layer of visible units
	 *
	 * @param num_visible_units Number of visible units
	 * @param unit_type Type of visible units
	 */
	CDeepBeliefNetwork(int32_t num_visible_units,
		ERBMVisibleUnitType unit_type = RBMVUT_BINARY);

	virtual ~CDeepBeliefNetwork();

	/** Adds a layer of hidden units. The layer is connected to the layer that
	 * was added directly before it.
	 *
	 * @param num_units Number of hidden units
	 */
	virtual void add_hidden_layer(int32_t num_units);

	/** Initializes the DBN
	 *
	 * @param sigma Standard deviation of the gaussian used to initialize the
	 * weights
	 */
	virtual void initialize(float64_t sigma = 0.01);

	/** Sets the number of train/test cases the RBM will deal with
	 *
	 * @param batch_size Batch size
	 */
	virtual void set_batch_size(int32_t batch_size);

	/** Pre-trains the DBN as a stack of RBMs
	 *
	 * @param features Input features. Should have as many features as the
	 * number of visible units in the DBN
	 */
	virtual void pre_train(CDenseFeatures<float64_t>* features);

	/** Pre-trains a single RBM
	 *
	 * @param index Index of the RBM
	 * @param features Input features. Should have as many features as the total
	 * number of visible units in the DBN
	 */
	virtual void pre_train(int32_t index, CDenseFeatures<float64_t>* features);

	/** Trains the DBN using the variant of the wake-sleep algorithm described
	 * in [A Fast Learning Algorithm for Deep Belief Nets, Hinton, 2006].
	 *
	 * @param features Input features. Should have as many features as the total
	 * number of visible units in the DBN
	 */
	virtual void train(CDenseFeatures<float64_t>* features);

	/** Applies the DBN as a features transformation
	 *
	 * Forward-propagates the input features through the DBN and returns the
	 * Mean activations of the \f$ i^{th} \f$ hidden layer
	 *
	 * @param features Input features. Should have as many features as the
	 * number of visible units in the DBN
	 * @param i Index of the hidden layer. If -1, the activations of the last
	 * hidden layer is returned
	 *
	 * @return Mean activations of the \f$ i^{th} \f$ hidden layer
	 */
	virtual CDenseFeatures<float64_t>* transform(
		CDenseFeatures<float64_t>* features, int32_t i=-1);

	/** Draws samples from the marginal distribution of the visible units. The
	 * sampling starts from the values DBN's internal state and result of the
	 * sampling is stored there too.
	 *
	 * @param num_gibbs_steps Number of Gibbs sampling steps for the top RBM.
	 * @param batch_size Number of samples to be drawn. A seperate chain is used
	 * for each sample
	 *
	 * @return Sampled states of the visible units
	 */
	virtual CDenseFeatures<float64_t>* sample(
		int32_t num_gibbs_steps=1, int32_t batch_size=1);

	/** Resets the state of the markov chain used for sampling */
	virtual void reset_chain();

	/** Converts the DBN into a neural network with the same structure and
	 * parameters. The visible layer in the DBN is converted into a
	 * CNeuralInputLayer object, and the hidden layers are converted into
	 * CNeuralLogisticLayer objects. An output layer can also be stacked on top
	 * of the last hidden layer
	 *
	 * @param output_layer An output layer
	 * @param sigma Standard deviation of the gaussian used to initialize the
	 * parameters of the output layer
	 *
	 * @return Neural network inititialized using the DBN
	 */
	virtual CNeuralNetwork* convert_to_neural_network(
		CNeuralLayer* output_layer=NULL, float64_t sigma = 0.01);

	/** Returns the weights matrix between layer i and i+1
	 *
	 * @param index Layer index
	 * @param p If specified, the weight matrix is extracted from it instead of
	 * m_params
	 */
	virtual SGMatrix<float64_t> get_weights(int32_t index,
		SGVector<float64_t> p = SGVector<float64_t>());

	/** Returns the bias vector of layer i
	 *
	 * @param index Layer index
	 * @param p If specified, the bias vector is extracted from it instead of
	 * m_params
	 */
	virtual SGVector<float64_t> get_biases(int32_t index,
		SGVector<float64_t> p = SGVector<float64_t>());

	virtual const char* get_name() const { return "DeepBeliefNetwork"; }

protected:
	/** Computes the states of some layer using the states of the layer above it */
	virtual void down_step(int32_t index, SGVector<float64_t> params,
		SGMatrix<float64_t> input, SGMatrix<float64_t> result,
		bool sample_states = true);

	/** Computes the states of some layer using the states of the layer below it */
	virtual void up_step(int32_t index, SGVector<float64_t> params,
		SGMatrix<float64_t> input, SGMatrix<float64_t> result,
		bool sample_states = true);

	/** Computes the gradients using the wake-sleep algorithm */
	virtual void wake_sleep(SGMatrix<float64_t> data,
		CRBM* top_rbm,
		SGMatrixList<float64_t> sleep_states,
		SGMatrixList<float64_t> wake_states,
		SGMatrixList<float64_t> psleep_states,
		SGMatrixList<float64_t> pwake_states,
		SGVector<float64_t> gen_params,
		SGVector<float64_t> rec_params,
		SGVector<float64_t> gen_gradients,
		SGVector<float64_t> rec_gradients);

private:
	void init();

public:
	/** CRBM::cd_num_steps for pre-training each RBM.
	 * Default value is 1 for all RBMs
	 */
	SGVector<int32_t> pt_cd_num_steps;

	/** CRBM::cd_persistent for pre-training each RBM.
	 * Default value is true for all RBMs
	 */
	SGVector<bool> pt_cd_persistent;

	/** CRBM::cd_sample_visible for pre-training each RBM.
	 * Default value is false for all RBMs
	 */
	SGVector<bool> pt_cd_sample_visible;

	/** CRBM::l2_coefficient for pre-training each RBM.
	 * Default value is 0.0 for all RBMs
	 */
	SGVector<float64_t> pt_l2_coefficient;

	/** CRBM::l1_coefficient for pre-training each RBM.
	 * Default value is 0.0 for all RBMs
	 */
	SGVector<float64_t> pt_l1_coefficient;

	/** CRBM::monitoring_interval for pre-training each RBM.
	 * Default value is 10 for all RBMs
	 */
	SGVector<int32_t> pt_monitoring_interval;

	/** CRBM::monitoring_method for pre-training each RBM.
	 * Default value is RBMMM_RECONSTRUCTION_ERROR for all RBMs
	 */
	SGVector<int32_t> pt_monitoring_method;

	/** CRBM::max_num_epochs for pre-training each RBM.
	 * Default value is 1 for all RBMs
	 */
	SGVector<int32_t> pt_max_num_epochs;

	/** CRBM::gd_mini_batch_size for pre-training each RBM.
	 * Default value is 0 for all RBMs
	 */
	SGVector<int32_t> pt_gd_mini_batch_size;

	/** CRBM::gd_learning_rate for pre-training each RBM.
	 * Default value is 0.1 for all RBMs
	 */
	SGVector<float64_t> pt_gd_learning_rate;

	/** CRBM::gd_learning_rate_decay for pre-training each RBM.
	 * Default value is 1.0 for all RBMs
	 */
	SGVector<float64_t> pt_gd_learning_rate_decay;

	/** CRBM::gd_momentum for pre-training each RBM.
	 * Default value is 0.9 for all RBMs
	 */
	SGVector<float64_t> pt_gd_momentum;

	/** Number of Gibbs sampling steps performed before each weight update during
	 * wake-sleep training. Default value is 1.
	 */
	int32_t cd_num_steps;

	/** Number of weight updates between each evaluation of the reconstruction
	 * error during wake-sleep training. Default value is 10.
	 */
	int32_t monitoring_interval;

	/** Maximum number of iterations over the training set during wake-sleep training.
	 * Defualt value is 1
	 */
	int32_t max_num_epochs;

	/** Size of the mini-batch used during gradient descent wake-sleep training,
	 * If 0 full-batch training is performed
	 * Default value is 0
	 */
	int32_t gd_mini_batch_size;

	/** Gradient descent learning rate for wake-sleep training, defualt value 0.1 */
	float64_t gd_learning_rate;

	/** Gradient descent learning rate decay for wake-sleep training.
	 * The learning rate is updated at each iteration i according to:
	 * alpha(i)=decay*alpha(i-1)
	 * Default value is 1.0 (no decay)
	 */
	float64_t gd_learning_rate_decay;

	/** gradient descent momentum multiplier for wake-sleep training
	 *
	 * default value is 0.9
	 *
	 * For more details on momentum, see this
	 * [paper](http://jmlr.org/proceedings/papers/v28/sutskever13.html)
	 * [Sutskever, 2013]
	 */
	float64_t gd_momentum;

protected:
	/** Type of the visible units */
	ERBMVisibleUnitType m_visible_units_type;

	/** Number of layers */
	int32_t m_num_layers;

	/** Size of each layer */
	CDynamicArray<int32_t>* m_layer_sizes;

	/** States of each layer */
	SGMatrixList<float64_t> m_states;

	/** Number of train/test cases the network is currently dealing with */
	int32_t m_batch_size;

	/** Parameters of the network */
	SGVector<float64_t> m_params;

	/** Number of parameters */
	int32_t m_num_params;

	/** Index at which the bias of each layer is stored in the parameters vector */
	SGVector<int32_t> m_bias_index_offsets;

	/** Index at which the weights of each hidden layer is stored in the
	 * parameters vector
	 */
	SGVector<int32_t> m_weights_index_offsets;

	/** Standard deviation of the gaussian used to initialize the
	 * parameters */
	float64_t m_sigma;
};

}
#endif
#endif
