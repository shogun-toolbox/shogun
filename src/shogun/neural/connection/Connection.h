/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#ifndef _CONNECTION_H__
#define _CONNECTION_H__

#ifdef HAVE_EIGEN3

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

#include <shogun/neural/layer/NeuralLayer.h>

namespace shogun
{

/** @brief Class Connection represents the connection between layers in neural
 * network.
 */
class Connection : public CSGObject
{
public:
    /** constructor */
    Connection();
    
    /** getter for the source layer */
    CNeuralLayer* get_source_layer() const;
    
    /** setter for the source layer */
    void set_source_layer(CNeuralLayer *layer);
    
    /** getter for the dest layer */
    CNeuralLayer* get_dest_layer() const;
    
    /** setter for the dest layer */        
    void set_dest_layer(CNeuralLayer *layer);
    
    /** destructor */
    virtual ~Connection();
protected:
    /** Connection delay - usually used in reccurent neural networks */
    index_t m_delay;
    
    /** Layers, linked by this connection */
    CNeuralLayer *m_source_layer, *m_dest_layer;
    
    /** Connection's weight matrix */
    SGMatrix<float64_t> m_weights_matrix;
}; //Connection class
    
} // shogun namespace

#endif // HAVE_EIGEN3

#endif // _CONNECTION_H__
