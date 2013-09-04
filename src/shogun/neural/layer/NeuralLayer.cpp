/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#include "shogun/neural/layer/NeuralLayer.h"

#include <shogun/neural/connection/Connection.h>

#ifdef HAVE_EIGEN3

using namespace shogun;

CNeuralLayer::CNeuralLayer(index_t neurons_num, CActivationFunction *activation_func)
{
    m_neurons_num = neurons_num;
    SG_REF(activation_func);
    m_activation_func = activation_func;
}

void CNeuralLayer::set_input(SGVector<float64_t> *input)
{
    ASSERT(input != NULL)
    ASSERT(input->size() == m_neurons_num)
    m_input = input;
}

void CNeuralLayer::add_to_input(const SGVector<float64_t>& vector)
{
    ASSERT(m_input != NULL)
    m_input->add(vector);
}

SGVector<float64_t> CNeuralLayer::get_output() const
{
    ASSERT(m_input != NULL)
    return get_output(*m_input);
}

SGVector<float64_t> CNeuralLayer::get_output(const SGVector<float64_t> &input) const
{
    ASSERT(input.size() == m_neurons_num)
    SGVector<float64_t> result(m_neurons_num);
    //TODO: parallelize
    for (index_t i=0; i < m_neurons_num; ++i)
    {
        result[i] = (*m_activation_func) (input[i]);
    }
    return result;
}

SGVector<float64_t> CNeuralLayer::get_first_derivative() const
{
    ASSERT(m_input != NULL)
    return get_first_derivative(*m_input);
}

SGVector<float64_t> CNeuralLayer::get_first_derivative
    (const SGVector<float64_t> &input) const
{
    ASSERT(input.size() == m_neurons_num)
    SGVector<float64_t> result(m_neurons_num);
    //TODO: parallelize
    for (index_t i=0; i < m_neurons_num; ++i)
    {
        result[i] = m_activation_func->get_first_derivative(input[i]);
    }
    return result;
}

CActivationFunction* CNeuralLayer::get_activation_function() const
{
    SG_REF(m_activation_func);
    return m_activation_func;
}

void CNeuralLayer::add_connection(const Connection *connection)
{
    ASSERT(connection->get_source_layer() == this)
    m_outgoing_connections.add(connection);
}

CNeuralLayer::~CNeuralLayer()
{
}

#endif //HAVE_EIGEN3
