/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang
 */

#ifndef ECOCUTIL_H__
#define ECOCUTIL_H__

#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>

namespace shogun
{

/** A helper class for some ECOC related procedures. */
class ECOCUtil
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
                dist += static_cast<int32_t>(Math::abs((c1[i]-c2[i])));
            return dist/2;
        }
};

} /* shogun */

#endif /* end of include guard: ECOCUTIL_H__ */

