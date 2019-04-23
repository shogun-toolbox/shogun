/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Bjoern Esser, Chiyuan Zhang
 */

#ifndef ECOCFORESTENCODER_H__
#define ECOCFORESTENCODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCDiscriminantEncoder.h>

namespace shogun
{

/** Forest ECOC Encoder.
 *
 * A data-dependent ECOC coding scheme that learns a tree-style codebook. See the
 * following paper for details
 *
 *   Sergio Escalera, Oriol Pujol, Petia Radeva. Boosted Landmarks of
 *   Contextual Descriptors and Forest-ECOC: A novel framework to detect and
 *   classify objects in cluttered scenes. Pattern Recognition Letters, 2007.
 *
 */
class ECOCForestEncoder: public ECOCDiscriminantEncoder
{
public:
    /** constructor */
    ECOCForestEncoder();

    /** destructor */
    virtual ~ECOCForestEncoder() {}

    /** get name */
    virtual const char* get_name() const { return "ECOCForestEncoder"; }

    /** get number of trees */
    int32_t get_num_trees() const { return m_num_trees; }

    /** set number of trees */
    void set_num_trees(int32_t num_trees);
};

} /*  shogun */

#endif /* end of include guard: ECOCFORESTENCODER_H__ */

