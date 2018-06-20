/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Elfarouk, Khaled Nasr
 */

#ifndef __STANNEURALNETWORK_H__
#define __STANNEURALNETWORK_H__

#include <shogun/optimization/StanFirstOrderSAGCostFunction.h>
#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{
template<class T> class CDenseFeatures;
class CDynamicObjectArray;
class CNeuralLayer;

/** @brief A generic multi-layer neural network
 *
 * A [Neural network](http://en.wikipedia.org/wiki/Artificial_neural_network)
 * is constructed using an array of StanNeuralLayer objects. The NeuralLayer
 * class defines the interface necessary for forward propagation only.
 *
 * The network can be constructed as any arbitrary directed acyclic graph.
 *
 * How to use the network:
 * 	- Prepare a CDynamicObjectArray of StanNeuralLayer-based objects that specify
 * the type of layers used in the network. The array must contain at least one
 * input layer. The last layer in the array is treated as the output layer.
 * Also note that forward propagation is performed in the order at which the
 * layers appear in the array. So if layer j takes its input from layer i then
 * i must be less than j.
 * 	- Specify how the layers are connected together. This can be done using
 * either connect() or quick_connect().
 * 	- Call initialize_neural_network()
 *  - Train using an optimizer
 * 	- Apply the network using apply()
 *
 * Supported feature types: CDenseFeatures<float64_t>
 * Supported label types:
 * 	- CBinaryLabels
 * 	- CMulticlassLabels
 * 	- CRegressionLabels
 *
 * The neural network can be trained using any of the minimizers in the optimization module
 *
 * The network stores the parameters of all the  layers in a single array. This
 * makes it easy to train a network of any combination of arbitrary layer types
 * using any optimization method
 *
 * All the matrices the network (and related classes) deal with are in
 * column-major format
 *
 */
class StanNeuralNetwork : public CSGObject
{

public:
	/** default constuctor */
	StanNeuralNetwork();

	/** Sets the layers of the network
	 *
	 * @param layers An array of CNeuralLayer objects specifying the layers of
	 * the network. Must contain at least one input layer. The last layer in
	 * the array is treated as the output layer
	 */
	StanNeuralNetwork(CDynamicObjectArray* layers);

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
	virtual void initialize_neural_network(float64_t sigma = 0.01f);

	virtual ~CNeuralNetwork();

	/** apply machine to data in means of binary classification problem */
	virtual CBinaryLabels* apply_binary(CFeatures* data);
	/** apply machine to data in means of regression problem */
	virtual CRegressionLabels* apply_regression(CFeatures* data);
	/** apply machine to data in means of multiclass classification problem */
	virtual CMulticlassLabels* apply_multiclass(CFeatures* data);

	/** returns a copy of a layer's parameters array
	 *
	 * @param i index of the layer
	 */
	StanVector& get_layer_parameters(int32_t i);

	/** returns the totat number of parameters in the network */
	int32_t get_num_parameters() { return m_total_num_parameters; }

	/** return the network's parameter array */
	StanVector& get_parameters() { return m_params; }

	/** returns the number of inputs the network takes*/
	int32_t get_num_inputs() { return m_num_inputs; }

	/** returns the number of neurons in the output layer */
	int32_t get_num_outputs();

	/** Returns an array holding the network's layers */
	CDynamicObjectArray* get_layers();

	virtual const char* get_name() const { return "StanNeuralNetwork";}


protected:
	/** Applies forward propagation, computes the activations of each layer up
	 * to layer j
	 *
	 * @param data input features
	 * @param j layer index at which the propagation should stop. If -1, the
	 * propagation continues up to the last layer
	 *
	 * @return activations of the last layer
	 */
	virtual StanMatrix forward_propagate(CFeatures* data, int32_t j=-1);
	virtual StanMatrix forward_propagate(SGMatrix<float64_t> inputs, int32_t j=-1);

	/** returns a pointer to layer i in the network */
	StanNeuralLayer* get_layer(int32_t i);

private:
	void init();

	/** Returns the section of vector v that belongs to layer i */
	template<class T>
	StanVector get_section(StanVector& v, int32_t i);

protected:
	/** number of neurons in the input layer */
	int32_t m_num_inputs;

	/** number of layer */
	int32_t m_num_layers;

	/** network's layers */
	CDynamicObjectArray* m_layers;

	/** Describes the connections in the network: if there's a connection from
	 * layer i to layer j then m_adj_matrix(i,j) = 1.
	 */
	SGMatrix<bool> m_adj_matrix;

	/** total number of parameters in the network */
	int32_t m_total_num_parameters;

	/** array where all the parameters of the network are stored */
	StanVector m_params;

	/** offsets specifying where each layer's parameters and parameter
	 * gradients are stored, i.e layer i's parameters are stored at
	 * m_params + m_index_offsets[i]
	 */
	SGVector<int32_t> m_index_offsets;

	/** True if the network is currently being trained
	 * initial value is false
	 */
	bool m_is_training;
};

}
#endif
