/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCUTIL_H__
#define ECOCUTIL_H__

#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>

namespace shogun
{

/** A helper class for some ECOC related procedures. */
class CECOCUtil
{
public:
    /** compute hamming distance.
     *
     * code elements can be -1, +1 or 0, when 0 occurs, that position of code
     * is not counted in the distance.
     */
    template<typename T1, typename T2>
        static int32_t hamming_distance(T1 *c1, T2 *c2, int32_t len)
        {
            int32_t dist = 0;
            for (int32_t i=0; i < len; ++i)
                dist += static_cast<int32_t>(CMath::abs((c1[i]-c2[i])));
            return dist/2;
        }
};

} /* shogun */

#endif /* end of include guard: ECOCUTIL_H__ */

