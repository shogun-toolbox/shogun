/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#ifndef _TRAINER_H__
#define _TRAINER_H__

#ifdef HAVE_EIGEN3

#include <shogun/base/SGObject.h>
#include <shogun/features/Features.h>

#include <shogun/neural/NeuralNetwork.h>

namespace shogun
{

/** @brief Class Trainer serves as an interface for all concrete trainers.
 *
 * A trainer is an object, that performs training of a neural network using
 * some specific algorithm (backpropagation, rprop, evolutionary methods etc).
 */
class CTrainer : public CSGObject
{
public:
    virtual bool train_network(CNeuralNetwork *network, CFeatures *data=NULL)=0;
}; // CTrainer class

} // shogun namespace

#endif //HAVE_EIGEN3

#endif // _TRAINER_H__
