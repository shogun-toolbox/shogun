/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang
 */

#ifndef ECOCOVRENCODER_H__
#define ECOCOVRENCODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCEncoder.h>

namespace shogun
{

/** One-vs-Rest Encoder */
class ECOCOVREncoder: public ECOCEncoder
{
public:
    /** constructor */
    ECOCOVREncoder() {}

    /** destructor */
    virtual ~ECOCOVREncoder() {}

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCOVREncoder";
    }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);
};

}

#endif /* end of include guard: ECOCOVRENCODER_H__ */
