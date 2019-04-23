/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Thoralf Klein, Yuyu Zhang
 */

#ifndef ECOCENCODER_H__
#define ECOCENCODER_H__

#include <shogun/lib/config.h>

#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/**
 * @brief ECOCEncoder produce an ECOC codebook.
 *
 * Note: for easy of implementation, our codebook is column-based. E.g. the
 * code-length is L and there are K classes, then our codebook will be a
 * L-by-K matrix, where each column corresponds to the code for each of the
 * K classes.
 *
 * The elements in the codebook can be
 *
 * - +1: positive class
 * - -1: negative class
 * - 0: ignore this class
 */
class ECOCEncoder: public SGObject
{
public:
    /** constructor */
    ECOCEncoder() {}

    /** destructor */
    virtual ~ECOCEncoder() {}

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCEncoder";
    }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes)=0;
};

}

#endif /* end of include guard: ECOCENCODER_H__ */
