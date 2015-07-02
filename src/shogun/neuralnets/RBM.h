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

#ifndef __RBM_H__
#define __RBM_H__

#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DynamicArray.h>

namespace shogun
{
enum ERBMMonitoringMethod
{
	RBMMM_RECONSTRUCTION_ERROR=0,
	RBMMM_PSEUDO_LIKELIHOOD=1
};

enum ERBMVisibleUnitType
{
	RBMVUT_BINARY=0,
	RBMVUT_GAUSSIAN=1,
	RBMVUT_SOFTMAX=2
};

/** @brief A Restricted Boltzmann Machine
 *
 * An [RBM](http://deeplearning.net/tutorial/rbm.html) is an energy based
 * probabilistic model. It consists of two groups of variables: the visible
 * variables \f$ v \f$ and the hidden variables \f$ h \f$. The key assumption
 * that RBMs make is that the hidden units are conditionally independent given
 * the visible units, and vice versa.
 *
 * The energy function for
 * RBMs with binary visible units is defined as:
 * \f[
 * E(v,h) = - b^T v - c^T h - h^T Wv
 * \f]
 * and for RBMs with gaussian (linear) visible units:
 * \f[
 * E(v,h) = v^T v - b^T v - c^T h - h^T Wv
 * \f]
 *
 * where \f$ b \f$ is the bias vector for the visible units, \f$ c \f$ is the
 * bias vector for the hidden units, and \f$ W \f$ is the weight matrix.
 *
 * The probability distribution is defined through the energy fucntion as:
 * \f[
 * P(v,h) = \frac{exp(-E(v,h))}{\sum_{v,h} exp(-E(v,h))}
 * \f]
 *
 * The above definitions along with the independence assumptions result in the
 * following conditionals:
 * \f[ P(h=1|v) = \frac{1}{1+exp(-Wv-c)} \quad \text{for binary hidden units} \f]
 * \f[ P(v=1|h) = \frac{1}{1+exp(-W^T h-b)} \quad \text{for binary visible units} \f]
 * \f[ P(v|h) \sim \mathcal{N} (W^T h + b,1) \quad \text{for gaussian visible units} \f]
 *
 * Note that when using gaussian visible units, the inputs should be normalized
 * to have zero mean and unity standard deviation.
 *
 * This class supports having multiple types of visible units in the same RBM.
 * The visible units are divided into groups where each group can have its
 * own type. The hidden units however are just one group of binary units.
 *
 * Samples can be drawn from the model using
 * [Gibbs sampling](http://en.wikipedia.org/wiki/Gibbs_sampling).
 *
 * Training is done using contrastive divergence [Hinton, 2002] or persistent
 * contrastive divergence [Tieleman, 2008] (default).
 *
 * Training progress can be monitored using the reconstruction error (default),
 * which is the average squared difference between a training batch and the RBM's
 * reconstruction of it. The reconstruction is generated using one step of gibbs
 * sampling. Progress can also be monitored using the
 * [pseudo-log-likelihood](http://en.wikipedia.org/wiki/Pseudolikelihood) which
 * is an approximation to the log-likelihood. However, this is currently only
 * supported for binary visible units.
 *
 * The rows of the visible_state matrix are divided into groups, one for each
 * group of visible units. For example, if we have 3 groups of visible units:
 * group 0 with 10 units, group 1 with 5 units, and group 2 with 6 units, the
 * states of group 0 will be stored in visible_state[0:10,:], the states of
 * group 1 will stored in visible_state[10:15,:], and the states of group 2
 * will be stored in visible_state[15:21,:]. Note that the groups are numbered
 * by the order in which they where added to the RBM using add_visible_group()
 */
class CRBM : public CSGObject
{
friend class CDeepBeliefNetwork;

public:
	/** default constructor */
	CRBM();

	/** Constructs an RBM with no visible units. The visible units can be added
	 * later using add_visible_group()
	 *
	 * @param num_hidden Number of hidden units
	 */
	CRBM(int32_t num_hidden);

	/** Constructs an RBM with a single group of visible units
	 *
	 * @param num_hidden Number of hidden units
	 * @param num_visible Number of visible units
	 * @param visible_unit_type Type of the visible units
	 */
	CRBM(int32_t num_hidden, int32_t num_visible,
		ERBMVisibleUnitType visible_unit_type = RBMVUT_BINARY);

	virtual ~CRBM();

	/** Adds a group of visible units to the RBM
	 *
	 * @param num_units Number of visible units
	 * @param visible_unit_type Type of the visible units
	 */
	virtual void add_visible_group(int32_t num_units, ERBMVisibleUnitType unit_type);

	/** Initializes the weights of the RBM. Must be called after all the visible
	 * groups have been added, and before the RBM is used.
	 *
	 * @param sigma Standard deviation of the gaussian used to initialize the
	 * weights
	 */
	virtual void initialize_neural_net(float64_t sigma=0.01);

	/** Sets the number of train/test cases the RBM will deal with
	 *
	 * @param batch_size Batch size
	 */
	virtual void set_batch_size(int32_t batch_size);

	/** Trains the RBM
	 *
	 * @param features Input features. Should have as many features as there
	 * are visible units in the RBM.
	 */
	virtual void train(CDenseFeatures<float64_t>* features);

	/** Draws samples from the marginal distribution of the visible units using
	 * Gibbs sampling. The sampling starts from the values in the RBM's
	 * visible_state matrix and result of the sampling is stored there too.
	 *
	 * @param num_gibbs_steps Number of Gibbs sampling steps
	 * @param batch_size Number of samples to be drawn. A seperate chain is used
	 * for each sample
	 */
	virtual void sample(int32_t num_gibbs_steps=1, int32_t batch_size=1);

	/** Draws Samples from \f$ P(V) \f$ where \f$ V \f$  is one of the visible
	 * unit groups. The sampling starts from the values in the RBM's
	 * visible_state matrix and result of the sampling is stored there too.
	 *
	 * @param V Index of the visible unit group to be sampled
	 * @param num_gibbs_steps Number of Gibbs sampling steps
	 * @param batch_size Number of samples to be drawn. A seperate chain is used
	 * for each sample
	 *
	 * @return Sampled states of group V
	 */
	virtual CDenseFeatures<float64_t>* sample_group(
			int32_t V,
			int32_t num_gibbs_steps=1, int32_t batch_size=1);

	/** Draws Samples from \f$ P(V|E=evidence) \f$ where \f$ E \f$ is one of
	 * the visible unit groups and \f$ V \f$ is all the visible unit excluding
	 * the ones in group \f$ E \f$. The sampling starts from the values in the
	 * RBM's visible_state matrix and result of the sampling is stored there too.
	 *
	 * @param E Index of the evidence visible unit group
	 * @param evidence States of the evidence visible unit group
	 * @param num_gibbs_steps Number of Gibbs sampling steps
	 */
	virtual void sample_with_evidence(
			int32_t E, CDenseFeatures<float64_t>* evidence,
			int32_t num_gibbs_steps=1);

	/** Draws Samples from \f$ P(V|E=evidence) \f$ where \f$ E \f$ is one of
	 * the visible unit groups and \f$ V \f$ is another visible unit group.
	 * The sampling starts from the values in the RBM's visible_state matrix
	 * and result of the sampling is stored there too.
	 *
	 * @param V Index of the visible unit group to be sampled
	 * @param E Index of the evidence visible unit group
	 * @param evidence States of the evidence visible unit group
	 * @param num_gibbs_steps Number of Gibbs sampling steps
	 *
	 * @return Sampled states of group V
	 */
	virtual CDenseFeatures<float64_t>* sample_group_with_evidence(
			int32_t V,
			int32_t E, CDenseFeatures<float64_t>* evidence,
			int32_t num_gibbs_steps=1);

	/** Resets the state of the markov chain used for sampling, which is stored
	 * in the visible_state matrix, to random values
	 */
	virtual void reset_chain();

	/** Computes the average free energy on a given batch of visible unit states.
	 *
	 * The free energy for a vector \f$ v \f$ is defined as:
	 * \f[ F(v) = - log(\sum_h exp(-E(v,h)) \f]
	 *
	 * which yields the following (in vectorized form):
	 * \f[ F(v) = -b^T v - \sum log(1+exp(Wv+c))
	 * \quad \text{for binary visible units}\f]
	 * \f[ F(v) = \frac{1}{2} v^T v - b^T v - \sum log(1+exp(Wv+c))
	 * \quad \text{for gaussian visible units}\f]
	 *
	 * @param visible States of the visible units
	 * @param buffer A matrix of size num_hidden*batch_size. used as a buffer
	 * during computation. If not given, a new matrix is allocated and used as
	 * a buffer.
	 *
	 * @return Average free energy over the given batch
	 */
	virtual float64_t free_energy(SGMatrix<float64_t> visible,
			SGMatrix<float64_t> buffer = SGMatrix<float64_t>());

	/** Computes the gradients of the free energy function with respect to the
	 * RBM's parameters
	 *
	 * @param visible States of the visible units
	 * @param gradients Array in which the results are stored.
	 * Length get_num_parameters()
	 * @param positive_phase If true, the result vector is reset to zero and
	 * the gradients are added to it with a positive sign. If false, the
	 * result vector is not reset and the gradients are added to it with a
	 * negative sign. This is useful during contrastive divergence.
	 * @param hidden_mean_given_visible Means of the hidden states given the
	 * visible states. If not given, means will be computed by calling
	 * mean_hidden()
	 */
	virtual void free_energy_gradients(SGMatrix<float64_t> visible,
			SGVector<float64_t> gradients,
			bool positive_phase = true,
			SGMatrix<float64_t> hidden_mean_given_visible = SGMatrix<float64_t>());

	/** Computes the gradients using contrastive divergence
	 *
	 * @param visible_batch States of the visible units
	 * @param gradients Array in which the results are stored.
	 * Length get_num_parameters()
	 */
	virtual void contrastive_divergence(SGMatrix<float64_t> visible_batch,
		SGVector<float64_t> gradients);

	/** Computes the average reconstruction error which is defined as:
	 * \f[ E = \frac{1}{N} \sum_i (v_i - \widetilde{v})^2 \f]
	 * where \f$ \widetilde{v} \f$ is computed using one step of gibbs sampling
	 * and \f$ N \f$ is the batch size
	 *
	 * @return Average reconstruction error over the given batch
	 */
	virtual float64_t reconstruction_error(SGMatrix<float64_t> visible,
		SGMatrix<float64_t> buffer = SGMatrix<float64_t>());

	/** Computes an approximation to the pseudo-likelihood.
	 * See this [tutorial](http://deeplearning.net/tutorial/rbm.html)
	 * for more details. Only works with binary visible units
	 *
	 * @param visible States of the visible units
	 * @param buffer A matrix of size num_visible*batch_size. used as a buffer
	 * during computation. If not given, a new matrix is allocated and used as
	 * a buffer.
	 *
	 * @param return Approximation to the average pseudo-likelihood over the
	 * given batch
	 */
	virtual float64_t pseudo_likelihood(SGMatrix<float64_t> visible,
		SGMatrix<float64_t> buffer = SGMatrix<float64_t>());

	/** Returns the states of the visible unit as CDenseFeatures<float64_t> */
	virtual CDenseFeatures<float64_t>* visible_state_features()
	{
		return new CDenseFeatures<float64_t>(visible_state);
	}

	/** Returns the parameter vector of the RBM */
	virtual SGVector<float64_t> get_parameters() { return m_params; }

	/** Returns the weights matrix
	 *
	 * @param p If specified, the weight matrix is extracted from it instead of
	 * m_params
	 */
	virtual SGMatrix<float64_t> get_weights(
		SGVector<float64_t> p = SGVector<float64_t>());

	/** Returns the bias vector of the hidden units
	 *
	 * @param p If specified, the bias vector is extracted from it instead of
	 * m_params
	 */
	virtual SGVector<float64_t> get_hidden_bias(
		SGVector<float64_t> p = SGVector<float64_t>());

	/** Returns the bias vector of the visible units
	 *
	 * @param p If specified, the bias vector is extracted from it instead of
	 * m_params
	 */
	virtual SGVector<float64_t> get_visible_bias(
		SGVector<float64_t> p = SGVector<float64_t>());

	/** Returns the number of parameters */
	virtual int32_t get_num_parameters() { return m_num_params; }

	virtual const char* get_name() const { return "RBM"; }

protected:
	/** Computes the mean of the hidden states given the visible states */
	virtual void mean_hidden(SGMatrix<float64_t> visible, SGMatrix<float64_t> result);

	/** Computes the mean of the visible states given the hidden states */
	virtual void mean_visible(SGMatrix<float64_t> hidden, SGMatrix<float64_t> result);

	/** Samples the hidden states according to the provided means */
	virtual void sample_hidden(SGMatrix<float64_t> mean, SGMatrix<float64_t> result);

	/** Samples the visible states according to the provided means */
	virtual void sample_visible(SGMatrix<float64_t> mean, SGMatrix<float64_t> result);

	/** Samples one group of visible states according to the provided means */
	virtual void sample_visible(int32_t index,
			SGMatrix<float64_t> mean, SGMatrix<float64_t> result);

private:
	void init();

public:
	/** Number of Gibbs sampling steps performed before each weight update during
	 * training. Default value is 1.
	 */
	int32_t cd_num_steps;

	/** If true, persistent contrastive divergence is used. Default value is true.
	 */
	bool cd_persistent;

	/** If true, the visible units are sampled during contrastive divergence. If
	 * false, the visible units are not sampled, and their mean values are used
	 * instead. Default value is false
	 */
	bool cd_sample_visible;

	/** L2 Regularization coeff, default value is 0.0*/
	float64_t l2_coefficient;

	/** L1 Regularization coeff, default value is 0.0*/
	float64_t l1_coefficient;

	/** Number of weight updates between each evaluation of the monitoring
	 * method. Default value is 10.
	 */
	int32_t monitoring_interval;

	/** Monitoring method */
	ERBMMonitoringMethod monitoring_method;

	/** maximum number of iterations over the training set.
	 * defualt value is 1
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

	/** States of the hidden units */
	SGMatrix<float64_t> hidden_state;

	/** States of the visible units */
	SGMatrix<float64_t> visible_state;

protected:
	/** Number of hidden units */
	int32_t m_num_hidden;

	/** Number of visible units */
	int32_t m_num_visible;

	/** Batch size */
	int32_t m_batch_size;

	/** Number of visible unit groups */
	int32_t m_num_visible_groups;

	/** Type of each visible unit group */
	CDynamicArray<int32_t>* m_visible_group_types;

	/** Size of each visible unit group */
	CDynamicArray<int32_t>* m_visible_group_sizes;

	/** Row offsets for accessing the states of each visible unit groups */
	CDynamicArray<int32_t>* m_visible_state_offsets;

	/** Number of parameters */
	int32_t m_num_params;

	/** Parameters */
	SGVector<float64_t> m_params;
};

}
#endif
#endif
