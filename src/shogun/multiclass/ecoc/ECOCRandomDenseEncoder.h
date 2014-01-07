/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCRANDOMDENSEENCODER_H__
#define ECOCRANDOMDENSEENCODER_H__

#include <multiclass/ecoc/ECOCEncoder.h>

namespace shogun
{

/** Generate random ECOC codebook containing +1 and -1, and
 * select the best one.
 */
class CECOCRandomDenseEncoder: public CECOCEncoder
{
public:
    /** constructor
     * @param maxiter max number of iterations
     * @param codelen code length, if set to zero, will be computed automatically via get_default_code_length
     * @param pposone probability of +1
     *
     * @see get_default_code_length
     */
    CECOCRandomDenseEncoder(int32_t maxiter=10000, int32_t codelen=0, float64_t pposone=0.5);

    /** destructor */
    virtual ~CECOCRandomDenseEncoder() {}

    /** set probability
     * @param pposone probability of +1
     */
    void set_probability(float64_t pposone);

    /** get name */
    virtual const char* get_name() const { return "ECOCRandomDenseEncoder"; }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);

    /** get default code length
     * @param num_classes number of classes
     *
     * In Dense Random Coding, 10 * log(num_classes) is suggested as code length.
     * See
     *
     *   E. Allwein, R. Schapire, and Y. Singer. Reducing multiclass to binary: A unifying approach
     *   for margin classifiers. Journal of Machine Learning Research, 1:113-141, 2002.
     */
    int32_t get_default_code_length(int32_t num_classes) const
    {
        return static_cast<int32_t>(CMath::round(10 * CMath::log(num_classes)));
    }

protected:
    int32_t   m_maxiter; ///< max number of iterations
    int32_t   m_codelen; ///< code length
    float64_t m_pposone; ///< probability of +1

private:
    /** ensure probability sum to one
     * @param pposone probability of +1
     */
    bool check_probability(float64_t pposone)
    {
        if (pposone >= 0.999 || pposone <= 0.0001)
            return false;
        return true;
    }

    /** init parameters */
    void init();
};

} // namespace shogun

#endif /* end of include guard: ECOCRANDOMDENSEENCODER_H__ */

