/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#ifndef _NEURALLAYER_H__
#define _NEURALLAYER_H__

#ifdef HAVE_EIGEN3

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/Set.h>

#include <shogun/neural/functions/ActivationFunction.h>

namespace shogun
{
    
class Connection;

/** @brief Class CNeuralLayer represents a set of neurons that share same properties
 * 
 * This is the most basic NeuralLayer class
 */
class CNeuralLayer : public CSGObject
{
public:
    /** default constructor */
    CNeuralLayer() : CSGObject() {}

    /** constructor
     *
	 * @param neurons_num number of neurons in the NeuralLayer
	 * @param activation_func type of neurons' activation function
     */
    CNeuralLayer(index_t neurons_num, CActivationFunction *activation_func);
    
    /** destructor */
    virtual ~CNeuralLayer();

    /** set NeuralLayer's input vector
     *
     * @param input vector of input values (input.size() == NeuralLayer.neurons_num)
     */
    void set_input(SGVector<float64_t> *input);
    
    /** add vector to the NeuralLayer's input vector
     *
     * @param vector vector of values, that will be added to the NeuralLayer's input
     */
    void add_to_input(const SGVector<float64_t>& vector);

    /** add connection to the set of connections, going out from the NeuralLayer
     *
     * @param connection connection to add
     */
    void add_connection(const Connection *connection);

    /** get vector of outputs. NeuralLayer's inputs must be set before - otherwise a
     * ShogunException will be raised.
     *
     * @return vector of outputs
     */
    SGVector<float64_t> get_output() const;
    
    /** get vector of outputs using passed in vector of inputs
     *
     * @param input vector of input values (input.size() == NeuralLayer.neurons_num)
     *
     * @return vector of outputs
     */
    virtual SGVector<float64_t> get_output(const SGVector<float64_t> &input)
    const;

    /** get vector of derivatives. NeuralLayer's inputs must be set before - otherwise
     * a ShogunException will be raised.
     *
     * @return vector of derivatives
     */
    SGVector<float64_t> get_first_derivative() const;

    /** get vector of derivatives using passed in vector of inputs
     *
     * @param input vector of input values (input.size() == NeuralLayer.neurons_num)
     *
     * @return vector of derivatives
     */
    virtual SGVector<float64_t> get_first_derivative
            (const SGVector<float64_t> &input) const;
    
    inline index_t get_neurons_num() const {return m_neurons_num; }

    /** getter for m_activation_func field */
    CActivationFunction* get_activation_function() const;
    
    const char* get_name() const {return "NeuralLayer";}

protected:
    /** Number of neurons in the NeuralLayer */
    index_t m_neurons_num;
    
    /** Activation function of all the neurons in the NeuralLayer */
    CActivationFunction *m_activation_func;
    
    /** NeuralLayer's input vector */
    SGVector<float64_t> *m_input;
    
    /** Set of connections, going out from NeuralLayer */
    CSet<const Connection*> m_outgoing_connections;
};

} // shogun namespace

#endif //HAVE_EIGEN3

#endif // _NEURALLAYER_H__
