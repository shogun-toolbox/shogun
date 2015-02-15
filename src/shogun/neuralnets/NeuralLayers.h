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
 * Written (W) 2014 Sergey Lisitsyn
 */

#ifndef __NEURALLAYERS_H__
#define __NEURALLAYERS_H__

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/neuralnets/NeuralLayer.h>

namespace shogun
{

/** @brief A class to construct neural layers.
 *
 * Uses builder-style syntax to describe layers
 * one by one.
 *
 *
 */
class CNeuralLayers : public CSGObject
{
public:
	/** default constructor */
	CNeuralLayers();

	/** destructor */
	virtual ~CNeuralLayers();

	/** Adds input neural layer. 
	 *
	 * @ref CNeuralInputLayer
	 *
	 * @param size the size of input layer
	 */
	CNeuralLayers* input(int32_t size);

	/** Adds logistic neural layer.
	 *
	 * @ref CNeuralLogisticLayer
	 *
	 * @param size the size of logistic layer
	 */
	CNeuralLayers* logistic(int32_t size);

	/** Adds linear neural layer.
	 *
	 * @ref CNeuralLinearLayer
	 *
	 * @param size the size of linear layer
	 */
	CNeuralLayers* linear(int32_t size);

	/** Adds rectified linear neural layer.
	 *
	 * @ref CNeuralRectifiedLinearLayer
	 *
	 * @param size the size of rectified linear layer
	 */
	CNeuralLayers* rectified_linear(int32_t size);

	/** Adds leaky rectified linear neural layer.
	 *
	 * @ref CNeuralLeakyRectifiedLinearLayer
	 *
	 * @param size the size of leaky rectified linear layer
	 */
	CNeuralLayers* leaky_rectified_linear(int32_t size);

	/** Adds softmax neural layer.
	 *
	 * @ref CNeuralSoftmaxLayer
	 *
	 * @param size the size of softmax layer
	 */
	CNeuralLayers* softmax(int32_t size);

	/** Adds custom neural layer.
	 *
	 * @param layer layer to add
	 */
	CNeuralLayers* with_layer(CNeuralLayer* layer);

	/** Finalizes 
	 */
	CDynamicObjectArray* done();

	/** Clears the constructed layers. 
	 */
	void clear();

	/** Returns true if there are no layers yet.
	 */
	bool empty();

	/** Returns name of the object
	 */
	virtual const char* get_name() const;

private:
	CDynamicObjectArray* m_layers;
};

}
#endif
