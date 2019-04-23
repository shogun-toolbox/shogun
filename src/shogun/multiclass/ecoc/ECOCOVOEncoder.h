/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuyu Zhang, Chiyuan Zhang
 */

#ifndef ECOCOVOENCODER_H__
#define ECOCOVOENCODER_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCEncoder.h>

namespace shogun
{

/** One-vs-One Encoder */
class ECOCOVOEncoder: public ECOCEncoder
{
public:
    /** constructor */
    ECOCOVOEncoder() {}

    /** destructor */
    virtual ~ECOCOVOEncoder() {}

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCOVOEncoder";
    }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);
};

}

#endif /* end of include guard: ECOCOVOENCODER_H__ */
