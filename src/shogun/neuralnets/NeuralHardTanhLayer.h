/*
 * Copyright (c) 2016 Shogun Toolbox Foundation
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
 * Written (W) 2016 Arasu Arun
 */

#ifndef __NEURALHARDTANHLAYER_H__
#define __NEURALHARDTANHLAYER_H__

#include <shogun/neuralnets/NeuralLinearLayer.h>

namespace shogun
{
/** @brief Neural layer with [hard tanh neurons]
 * (http://ronan.collobert.com/pub/matos/2004_links_icml.pdf)
 *
 * Activations are computed according to:
 * 	\f[ 
 * 	\begin{cases} 
 * 		max_act &\mbox{if } z > max_act \\ 
 * 		min_act &\mbox{if } z <  min_act \\
 * 		z& \mbox{otherwise}  
 * 	\end{cases}
 * 	\f]
 * where 
 * \f[ z=W*x+b \f] 
 * and W is the weight matrix, b is the bias vector, x is the input vector, 
 * and min_act, max_act are parameters.
 * Default value of min_act is -1.0 and max_act is 1.0.
 *
 * When used as an output layer, a squared error measure is used
 */
class CNeuralHardTanhLayer : public CNeuralLinearLayer
{
public:
	/** default constructor */
	CNeuralHardTanhLayer();

	/** Constuctor
	 *
	 * @param num_neurons Number of neurons in this layer
	 */
	CNeuralHardTanhLayer(int32_t num_neurons);

	virtual ~CNeuralHardTanhLayer() {}

	/** Sets the lower bound of the activation value
	 *
	 * @param min_act new value of min_act
	 */
	virtual void set_min_act(float64_t min_act) { m_min_act=min_act; }

	/** Gets the lower bound of the activation value
	 *
	 * @return min_act
	 */
	virtual float64_t get_min_act() { return m_min_act; }

	/** Sets the upper bound of the activation value
	 *
	 * @param max_act new value of max_act
	 */
	virtual void set_max_act(float64_t max_act) { m_max_act=max_act; }

	/** Gets the upper bound of the activation value
	 *
	 * @return max_act
	 */
	virtual float64_t get_max_act() { return m_max_act; }

	/** Computes the activations of the neurons in this layer, results should
	 * be stored in m_activations. To be used only with non-input layers
	 *
	 * @param parameters Vector of size get_num_parameters(), contains the
	 * parameters of the layer
	 *
	 * @param layers Array of layers that form the network that this layer is
	 * being used with
	 *
	 */
	virtual void compute_activations(SGVector<float64_t> parameters,
		CDynamicObjectArray* layers);	

	virtual const char* get_name() const { return "NeuralHardTanhLayer"; }

private:
	void init();

protected:
	/** Parameter used to set lower bound of activation value
	 */
	float64_t m_min_act;

	/** Parameter used to set upper bound of activation value
	 */
	float64_t m_max_act;
};

}
#endif //__NEURALHARDTANHLAYER_H__
