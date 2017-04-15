/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#include "shogun/neural/NeuralNetwork.h"

#ifdef HAVE_EIGEN3

#include <set>

#include <shogun/base/SGObject.h>

#include "shogun/neural/trainer/Trainer.h"
#include "shogun/neural/layer/NeuralLayer.h"
#include "shogun/neural/connection/Connection.h"

namespace shogun
{

/** @brief Class NetworkImplementation is responsible for storing and
 * working with inner implementation of a neural network. Is not supposed
 * to be inherited.
 */
class NetworkImplementation : public CSGObject
{
public:
    /** add new layer to the network
     *
     * @param new_layer layer to add
     *
     * @return if new layer was added
     */
    bool add_layer(CNeuralLayer *new_layer);

    /** add new connection to the network
     * If at least one of layers, connected by the given connection, is not
     * present in the network's layers set - ShogunException will be raised.
     *
     * @param new_connection connection to add
     *
     * @return if new connection was added
     */
    bool add_connection(Connection *new_connection);
    
    /** destructor */
    ~NetworkImplementation();
    
    const char* get_name() const { return "NetworkImplementation";}

private:
    /** A neural network is basically a set of interconnected layers of neurons
     * (a layer can consist of one neuron). Thus, it can be represented as a
     * graph, where each vertex represents network's layer, and edge represents
     * the connection between layers (there can possibly be different types of
     * connections<->edges). The usual graph representation is 2 sets - set of
     * vertices and set of edges.
     */
    std::set<CNeuralLayer*> m_layers;
    std::set<Connection*> m_connections;
    /** Each network should have an input and an output layer (one layer can be
     * input and output simultaneously).
     */
    CNeuralLayer *m_input_layer, *m_output_layer;
}; // NetworkImplementation class

} // shogun namespace

using namespace shogun;

bool NetworkImplementation::add_layer(CNeuralLayer *new_layer)
{
    bool wasInserted = this->m_layers.insert(new_layer).second;
    if (wasInserted)
        SG_REF(new_layer);
    return wasInserted;
}

bool NetworkImplementation::add_connection(Connection *new_connection)
{
    if (!m_layers.count(new_connection->get_source_layer()) ||
        !m_layers.count(new_connection->get_dest_layer()))
        /** trying to insert connection, that connects layers not present in
         * the graph - layers should be inserted before the connection.
         */
        SG_ERROR("Either destination or source layer of passed in Connection\
is not present in the neural network.\n");

    bool wasInserted = this->m_connections.insert(new_connection).second;
    if (wasInserted)
        SG_REF(new_connection);
    return wasInserted;
}

NetworkImplementation::~NetworkImplementation()
{
    typedef std::set<CNeuralLayer*>::iterator layer_iter;
    typedef std::set<Connection*>::iterator conn_iter;
    for (layer_iter it = m_layers.begin(); it != m_layers.end(); ++it)
    {
        CNeuralLayer *layer_to_free = *it;
        SG_UNREF(layer_to_free);
    }
    for (conn_iter it = m_connections.begin(); it != m_connections.end(); ++it)
    {
        Connection *connection_to_free = *it;
        SG_UNREF(connection_to_free);
    }
}

CNeuralNetwork::CNeuralNetwork()
: CMachine()
{
    init();
}

void CNeuralNetwork::init()
{
    m_network_impl = new NetworkImplementation();
    SG_REF(m_network_impl);
}

inline bool CNeuralNetwork::add_layer(CNeuralLayer *new_layer)
{
    return m_network_impl->add_layer(new_layer);
}

inline bool CNeuralNetwork::add_connection(Connection *new_connection)
{
    return m_network_impl->add_connection(new_connection);
}

inline CTrainer* CNeuralNetwork::get_trainer() const
{
    SG_REF(m_trainer);
    return m_trainer;
}

inline void CNeuralNetwork::set_trainer(CTrainer *trainer)
{
    SG_REF(trainer);
    SG_UNREF(m_trainer);
    m_trainer = trainer;
}

bool CNeuralNetwork::train_machine(CFeatures *data)
{
    ASSERT(m_trainer != NULL)
    return m_trainer->train_network(this, data);
}

CNeuralNetwork::~CNeuralNetwork()
{
    SG_UNREF(m_trainer);
    SG_UNREF(m_network_impl);
}

#endif //HAVE_EIGEN3
