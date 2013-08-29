/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/mathematics/logdet/OperatorFunction.h>

using namespace shogun;

template class COperatorFunction<bool>;
template class COperatorFunction<char>;
template class COperatorFunction<int8_t>;
template class COperatorFunction<uint8_t>;
template class COperatorFunction<int16_t>;
template class COperatorFunction<uint16_t>;
template class COperatorFunction<int32_t>;
template class COperatorFunction<uint32_t>;
template class COperatorFunction<int64_t>;
template class COperatorFunction<uint64_t>;
template class COperatorFunction<float32_t>;
template class COperatorFunction<float64_t>;
template class COperatorFunction<floatmax_t>;
template class COperatorFunction<complex64_t>;
