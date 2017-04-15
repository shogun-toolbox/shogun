/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#ifndef _NEURALNETWORK_H__
#define _NEURALNETWORK_H__

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/Machine.h>
#include <shogun/features/Features.h>
#include <shogun/loss/LossFunction.h>

namespace shogun
{

class CTrainer;
class NetworkImplementation;
class CNeuralLayer;
class Connection;

/** @brief Class NeuralNetwork serves as a base class for all concrete types
 * of networks, thus it should be as general as possible.
 */
class CNeuralNetwork : public CMachine
{
public:
    /** constructor */
    CNeuralNetwork();

    /** destructor */
    virtual ~CNeuralNetwork();
    
    inline CTrainer* get_trainer() const;
    
    inline void set_trainer(CTrainer *trainer);
    
protected:
    /** train a neural network
     *
     * @param data training data
     * @return whether training was successful
     */
    virtual bool train_machine(CFeatures* data=NULL);

    /** Trainer - an object, that will perform actual training of the network.*/
    CTrainer *m_trainer;
    
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
    
    /** Inner implementation of the neural network */
    NetworkImplementation *m_network_impl;

    /** Loss function used by this neural network */
    CLossFunction *m_loss_function;
private:
    void init();
}; /* CNeuralNetwork class */

} /* shogun namespace */


#endif //HAVE_EIGEN3

#endif //_NEURALNETWORK_H__
