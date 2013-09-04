/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#include "shogun/neural/connection/Connection.h"

using namespace shogun;

#ifdef HAVE_EIGEN3

CNeuralLayer* Connection::get_source_layer() const
{
    SG_REF(m_source_layer);
    return m_source_layer;
}

void Connection::set_source_layer(CNeuralLayer *layer)
{
    SG_REF(layer);
    SG_UNREF(m_source_layer);
    m_source_layer = layer;
}

CNeuralLayer* Connection::get_dest_layer() const
{
    SG_REF(m_dest_layer);
    return m_dest_layer;
}

void Connection::set_dest_layer(CNeuralLayer *layer)
{
    SG_REF(layer);
    SG_UNREF(m_dest_layer);
    m_dest_layer = layer;
}

Connection::~Connection()
{
    SG_UNREF(m_source_layer);
    SG_UNREF(m_dest_layer);
}
#endif // HAVE_EIGEN3