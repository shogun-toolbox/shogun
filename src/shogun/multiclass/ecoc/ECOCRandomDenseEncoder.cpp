/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <algorithm>
#include <limits>

#include <multiclass/ecoc/ECOCRandomDenseEncoder.h>
#include <multiclass/ecoc/ECOCUtil.h>

using namespace shogun;

CECOCRandomDenseEncoder::CECOCRandomDenseEncoder(int32_t maxiter, int32_t codelen, float64_t pposone)
    :m_maxiter(maxiter), m_codelen(codelen), m_pposone(pposone)
{
    if (!check_probability(pposone))
        SG_ERROR("invalid probability of +1")

    init();
}

void CECOCRandomDenseEncoder::init()
{
    SG_ADD(&m_maxiter, "maxiter", "max number of iterations", MS_NOT_AVAILABLE);
    SG_ADD(&m_codelen, "codelen", "code length", MS_NOT_AVAILABLE);
    SG_ADD(&m_pposone, "pposone", "probability of +1", MS_NOT_AVAILABLE);
}

void CECOCRandomDenseEncoder::set_probability(float64_t pposone)
{
    if (!check_probability(pposone))
        SG_ERROR("probability of 0, +1 and -1 must sum to one")

    m_pposone = pposone;
}

SGMatrix<int32_t> CECOCRandomDenseEncoder::create_codebook(int32_t num_classes)
{
    int32_t codelen = m_codelen;
    if (codelen <= 0)
        codelen = get_default_code_length(num_classes);


    SGMatrix<int32_t> best_codebook(codelen, num_classes, true);
    int32_t best_dist = 0;

    SGMatrix<int32_t> codebook(codelen, num_classes);
    int32_t n_iter = 0;
    while (true)
    {
        // fill codebook
        codebook.zero();
        for (int32_t i=0; i < codelen; ++i)
        {
            for (int32_t j=0; j < num_classes; ++j)
            {
                float64_t randval = CMath::random(0.0, 1.0);
                if (randval > m_pposone)
                    codebook(i, j) = -1;
                else
                    codebook(i, j) = +1;
            }
        }

        bool valid = true;
        for (int32_t i=0; i < codelen; ++i)
        {
            bool p1_occur = false, n1_occur = false;
            for (int32_t j=0; j < num_classes; ++j)
                if (codebook(i, j) == 1)
                    p1_occur = true;
                else if (codebook(i, j) == -1)
                    n1_occur = true;

            if (!p1_occur || !n1_occur)
            {
                valid = false;
                break;
            }
        }

        if (valid)
        {
            // see if this is a better codebook
            // compute the minimum pairwise code distance
            int32_t min_dist = std::numeric_limits<int32_t>::max();
            for (int32_t i=0; i < num_classes; ++i)
            {
                for (int32_t j=i+1; j < num_classes; ++j)
                {
                    int32_t dist = CECOCUtil::hamming_distance(codebook.get_column_vector(i),
                            codebook.get_column_vector(j), codelen);
                    if (dist < min_dist)
                        min_dist = dist;
                }
            }

            if (min_dist > best_dist)
            {
                best_dist = min_dist;
                std::copy(codebook.matrix, codebook.matrix + codelen*num_classes,
                        best_codebook.matrix);
            }
        }

        if (++n_iter >= m_maxiter)
            if (best_dist > 0) // already obtained a good codebook
                break;
    }

    return best_codebook;
}
