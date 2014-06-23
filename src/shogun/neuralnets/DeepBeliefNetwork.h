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

namespace shogun
{
template <class T> class SGVector;
template <class T> class SGMatrix;
template <class T> class CDenseFeatures;
template <class T> class CDynamicArray;
class CDynamicObjectArray;
class CNeuralNetwork;
class CNeuralLayer;

/** @brief A Deep Belief Network
 * 
 * A [Deep Belief Network](http://www.scholarpedia.org/article/Deep_belief_networks)
 * [Hinton, 2006] is a multilayer probabilistic generative models. It consists 
 * of hidden layers and visible layers. The top hidden layer and the layers that 
 * are connected to it form a Restricted Boltzmann Machine. The rest of 
 * connections in the network are directed connections that go from a hidden 
 * layer into a visible layer or another hidden layer.
 * 
 * The network can be pre-trained by treating it as a stack of RBMs. Each hidden 
 * layer, along with the layers that are connected to it, form an RBM. Each RBM 
 * is then trained using (persistent) contrastive divergence. Pre-training often 
 * provides a good initialization for the network's parameters. After pre-training,
 * the DBN can be used to initialize the parameters of a neural network.
 * 
 * Samples can be drawn from the model by starting with a random state for the 
 * top hidden layer, performing some steps of Gibbs sampling in the top RBM to 
 * obtain the states of the top hidden layer and the layers that are connected to 
 * it, and then using those to infer the states of the other layers.
 * 
 * Steps for using the DBN class:
 * 	- Construct a CDeepBeliefNetwork using the default constructor
 * 	- Add the visible and hidden layers using add_visible_layer() and 
 * add_hidden_layer(). Note that each layer can only have one outgoing connection.
 * 	- Initialize the DBN using initialize()
 * 	- Pre-train the DBN using pre_train()
 * 	- If needed, call convert_to_neural_network() to convert the DBN into a 
 * neural network
 * 
 */
class CDeepBeliefNetwork : public CSGObject
{
public:
	/** default constructor */
	CDeepBeliefNetwork();
	
	virtual ~CDeepBeliefNetwork();
	
	/** Add a layer of visible units to the DBN
	 * 
	 * @param num_units Number of visible units
	 * @param unit_type Type of visible units
	 * @param row_offset Index of the first feature that the layer connects to, 
	 * i.e during training, the states of the layer are copied from 
	 * input_features[row_offset:row_offset+num_units]
	 */
	virtual void add_visible_layer(int32_t num_units, 
			ERBMVisibleUnitType unit_type, int32_t row_offset=0);
	
	/** Adds a layer of hidden units. The layer is connected to the layer that 
	 * was added directly before it.
	 * 
	 * @param num_units Number of hidden units
	 */
	virtual void add_hidden_layer(int32_t num_units);
	
	/** Adds a layer of hidden units and connects it to another layer
	 * 
	 * @param num_units Number of hidden units
	 * @param connection_index Index of the layer that this layer is to be connected 
	 * to.
	 */
	virtual void add_hidden_layer(int32_t num_units, 
			int32_t connection_index);
	
	/** Adds a layer of hidden units and connects it to two other layer
	 * 
	 * @param num_units Number of hidden units
	 * @param connection1_index Index of a layer that this layer is to be connected 
	 * to.
	 * @param connection2_index Index of a layer that this layer is to be connected 
	 * to.
	 */
	virtual void add_hidden_layer(int32_t num_units, 
			int32_t connection1_index, int32_t connection2_index);
	
	/** Adds a layer of hidden units and connects it to other layers
	 * 
	 * @param num_units Number of hidden units
	 * @param connection_indices Indices of the layer that this layer is to be connected 
	 * to.
	 */
	virtual void add_hidden_layer(int32_t num_units, 
			SGVector<int32_t> connection_indices);
	
	/** Initializes the DBN
	 * 
	 * @param sigma Standard deviation of the gaussian used to initialize the 
	 * weights
	 */
	virtual void initialize(float64_t sigma = 0.01);
	
	/** Pre-trains the DBN as a stack of RBMs
	 * 
	 * @param features Input features. Should have as many features as the total 
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
	
	/** Applies the DBN as a features transformation
	 * 
	 * Forward-propagates the input features through the DBN and returns the 
	 * Mean activations of the top layer hidden units
	 * 
	 * @param features Input features. Should have as many features as the total 
	 * number of visible units in the DBN
	 *
	 * @return Mean activations of the top layer hidden units
	 */
	virtual CDenseFeatures<float64_t>* transform(CDenseFeatures<float64_t>* features);
	
	/** Converts the DBN into a neural network with the same structure and 
	 * parameters. The visible layers in the DBN are converted into 
	 * CNeuralInputLayer objects, and the hidden layers are converted into 
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
	
	/** Draws samples from the marginal distribution of the visible units. The
	 * sampling starts from the values DBN's internal state and result of the 
	 * sampling is stored there too.
	 * 
	 * @param num_gibbs_steps Number of Gibbs sampling steps for the top RBM.
	 * @param batch_size Number of samples to be drawn. A seperate chain is used 
	 * for each sample
	 */
	virtual void sample(int32_t num_gibbs_steps=1, int32_t batch_size=1);
	
	/** Draws Samples from \f$ P(V) \f$ where \f$ V \f$  is one of the visible 
	 * layers. The sampling starts from the values DBN's internal state and 
	 * result of the sampling is stored there too.
	 * 
	 * @param V Index of the visible layer to be sampled
	 * @param num_gibbs_steps Number of Gibbs sampling steps for the top RBM.
	 * @param batch_size Number of samples to be drawn. A seperate chain is used 
	 * for each sample
	 * 
	 * @return Sampled states of layer V
	 */
	virtual CDenseFeatures<float64_t>* sample_layer(
			int32_t V, 
			int32_t num_gibbs_steps=1, int32_t batch_size=1);
	
	/** Draws Samples from \f$ P(V|E=evidence) \f$ where \f$ E \f$ is one of 
	 * the visible layers and \f$ V \f$ is all the visible layers excluding 
	 * \f$ E \f$. The sampling starts from the values DBN's internal state and 
	 * result of the sampling is stored there too.
	 * 
	 * @param E Index of the evidence visible layer
	 * @param evidence States of the evidence visible layer
	 * @param num_gibbs_steps Number of Gibbs sampling steps for the top RBM.
	 */
	virtual void sample_with_evidence(
			int32_t E, CDenseFeatures<float64_t>* evidence,
			int32_t num_gibbs_steps=1);
	
	/** Draws Samples from \f$ P(V|E=evidence) \f$ where \f$ E \f$ is one of 
	 * the visible layers and \f$ V \f$ is another layer. The sampling starts 
	 * from the values DBN's internal state and result of the sampling is 
	 * stored there too.
	 * 
	 * @param V Index of the visible layer to be sampled
	 * @param E Index of the evidence visible layer
	 * @param evidence States of the evidence visible layer
	 * @param num_gibbs_steps Number of Gibbs sampling steps for the top RBM.
	 * 
	 * @return Sampled states of layer V
	 */
	virtual CDenseFeatures<float64_t>* sample_layer_with_evidence(
			int32_t V, 
			int32_t E, CDenseFeatures<float64_t>* evidence,
			int32_t num_gibbs_steps=1);
	
	/** Sets CRBM::cd_num_steps for all RBMs in the network*/
	virtual void set_cd_num_steps(int32_t cd_num_steps) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_cd_num_steps(i, cd_num_steps);
	}
	
	/** Sets CRBM::cd_num_steps for a single rbm */
	virtual void set_cd_num_steps(int32_t rbm_index, int32_t cd_num_steps)
	{
		get_rbm(rbm_index)->cd_num_steps = cd_num_steps;
	}

	/** Sets CRBM::cd_persistent for all RBMs in the network*/
	virtual void set_cd_persistent(bool cd_persistent) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_cd_persistent(i, cd_persistent);
	}
	
	/** Sets CRBM::cd_persistent for a single rbm */
	virtual void set_cd_persistent(int32_t rbm_index, bool cd_persistent)
	{
		get_rbm(rbm_index)->cd_persistent = cd_persistent;
	}

	/** Sets CRBM::cd_sample_visible for all RBMs in the network*/
	virtual void set_cd_sample_visible(bool cd_sample_visible) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_cd_sample_visible(i, cd_sample_visible);
	}
	
	/** Sets CRBM::cd_sample_visible for a single rbm */
	virtual void set_cd_sample_visible(int32_t rbm_index, bool cd_sample_visible)
	{
		get_rbm(rbm_index)->cd_sample_visible = cd_sample_visible;
	}

	/** Sets CRBM::l2_coefficient for all RBMs in the network*/
	virtual void set_l2_coefficient(float64_t l2_coefficient) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_l2_coefficient(i, l2_coefficient);
	}
	
	/** Sets CRBM::l2_coefficient for a single rbm */
	virtual void set_l2_coefficient(int32_t rbm_index, float64_t l2_coefficient)
	{
		get_rbm(rbm_index)->l2_coefficient = l2_coefficient;
	}

	/** Sets CRBM::l1_coefficient for all RBMs in the network*/
	virtual void set_l1_coefficient(float64_t l1_coefficient) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_l1_coefficient(i, l1_coefficient);
	}
	
	/** Sets CRBM::l1_coefficient for a single rbm */
	virtual void set_l1_coefficient(int32_t rbm_index, float64_t l1_coefficient)
	{
		get_rbm(rbm_index)->l1_coefficient = l1_coefficient;
	}

	/** Sets CRBM::monitoring_interval for all RBMs in the network*/
	virtual void set_monitoring_interval(int32_t monitoring_interval) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_monitoring_interval(i, monitoring_interval);
	}
	
	/** Sets CRBM::monitoring_interval for a single rbm */
	virtual void set_monitoring_interval(int32_t rbm_index, int32_t monitoring_interval)
	{
		get_rbm(rbm_index)->monitoring_interval = monitoring_interval;
	}

	/** Sets CRBM::monitoring_method for all RBMs in the network*/
	virtual void set_monitoring_method(ERBMMonitoringMethod monitoring_method) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_monitoring_method(i, monitoring_method);
	}
	
	/** Sets CRBM::monitoring_method for a single rbm */
	virtual void set_monitoring_method(int32_t rbm_index, ERBMMonitoringMethod monitoring_method)
	{
		get_rbm(rbm_index)->monitoring_method = monitoring_method;
	}

	/** Sets CRBM::max_num_epochs for all RBMs in the network*/
	virtual void set_max_num_epochs(int32_t max_num_epochs) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_max_num_epochs(i, max_num_epochs);
	}
	
	/** Sets CRBM::max_num_epochs for a single rbm */
	virtual void set_max_num_epochs(int32_t rbm_index, int32_t max_num_epochs)
	{
		get_rbm(rbm_index)->max_num_epochs = max_num_epochs;
	}

	/** Sets CRBM::gd_mini_batch_size for all RBMs in the network*/
	virtual void set_gd_mini_batch_size(int32_t gd_mini_batch_size) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_gd_mini_batch_size(i, gd_mini_batch_size);
	}
	
	/** Sets CRBM::gd_mini_batch_size for a single rbm */
	virtual void set_gd_mini_batch_size(int32_t rbm_index, int32_t gd_mini_batch_size)
	{
		get_rbm(rbm_index)->gd_mini_batch_size = gd_mini_batch_size;
	}

	/** Sets CRBM::gd_learning_rate for all RBMs in the network*/
	virtual void set_gd_learning_rate(float64_t gd_learning_rate) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_gd_learning_rate(i, gd_learning_rate);
	}
	
	/** Sets CRBM::gd_learning_rate for a single rbm */
	virtual void set_gd_learning_rate(int32_t rbm_index, float64_t gd_learning_rate)
	{
		get_rbm(rbm_index)->gd_learning_rate = gd_learning_rate;
	}

	/** Sets CRBM::gd_learning_rate_decay for all RBMs in the network*/
	virtual void set_gd_learning_rate_decay(float64_t gd_learning_rate_decay) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_gd_learning_rate_decay(i, gd_learning_rate_decay);
	}
	
	/** Sets CRBM::gd_learning_rate_decay for a single rbm */
	virtual void set_gd_learning_rate_decay(int32_t rbm_index, float64_t gd_learning_rate_decay)
	{
		get_rbm(rbm_index)->gd_learning_rate_decay = gd_learning_rate_decay;
	}

	/** Sets CRBM::gd_momentum for all RBMs in the network*/
	virtual void set_gd_momentum(float64_t gd_momentum) 
	{
		for (int32_t i=0; i<m_rbms->get_num_elements(); i++)
			set_gd_momentum(i, gd_momentum);
	}
	
	/** Sets CRBM::gd_momentum for a single rbm */
	virtual void set_gd_momentum(int32_t rbm_index, float64_t gd_momentum)
	{
		get_rbm(rbm_index)->gd_momentum = gd_momentum;
	}
	
	virtual const char* get_name() const { return "DeepBeliefNetwork"; }

protected:
	/** Packs the states of the layers that form an RBM's visible_state matrix 
	 * into a single matrix. The states of the visible layers 
	 * are copied from (inputs) and the states of the hidden layers are copied 
	 * from their RBM's hidden_state matrices. The result is stored in (visible_state)
	 * 
	 * @param index RBM's index
	 * @param inputs Inputs to the DBN
	 * @param visible_state Result
	 */
	virtual void pack_into_visible_state_matrix(
		int32_t index, SGMatrix<float64_t> inputs, 
		SGMatrix<float64_t> visible_state);
	
	/** Unpacks an RBM's visible_state matrix into the states of the hidden layers
	 * that are connected to it
	 * 
	 * @param index RBM's index
	 */
	virtual void unpack_from_visible_state_matrix(int32_t index);
	
private:
	void init();
	
	CRBM* get_rbm(int32_t index);
	
protected:
	/** Number of layer */
	int32_t m_num_layers;
	
	/** Type of each layer, -1 for hidden layers */
	CDynamicArray<int32_t>* m_layer_types;
	
	/** Size of each layer */
	CDynamicArray<int32_t>* m_layer_sizes;
	
	/** Row offset for accessing the inputs to the RBM */
	CDynamicArray<int32_t>* m_layer_input_offsets;
	
	/** Index of the RBM that each layer belongs to */
	SGVector<int32_t> m_layer_rbm_indices;
	
	/** Index of the RBM visible layer group that each visible layer belongs to.
	 * 0 for hidden layers
	 */
	SGVector<int32_t> m_layer_rbm_group_indices;
	
	/** Adjacency matrix of the DBN */
	SGMatrix<bool> m_adj_matrix;
	
	/** Array of RBMs that form DBN */
	CDynamicObjectArray* m_rbms;
};

}
#endif
#endif
