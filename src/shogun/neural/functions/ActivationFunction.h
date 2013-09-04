/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2013 Vladyslav Gorbatiuk
 */

#ifndef _ACTIVATIONFUNCTION_H__
#define _ACTIVATIONFUNCTION_H__

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/base/SGObject.h>

namespace shogun
{
    
/** @brief class CActivationFunction is a basic interface for all activation
 * functions used in artificial neurons.
 */
class CActivationFunction : public CSGObject
{
public:

    /** get function's value for a given input
     *
     * @param input function's input
     * @return corresponding function's value
     */
    virtual float64_t operator() (float64_t input) const = 0;
    
    /** get function's first derivative for a given input
     * @param input function's input
     * @return corresponding value of first function's derivative
     */
    virtual float64_t get_first_derivative(float64_t input) const = 0;

    const char* get_name() const { return "ActivationFunction"; }
}; // CActivationFunction class

} // shogun namespace

#endif // HAVE_EIGEN3

#endif // _ACTIVATIONFUNCTION_H__
