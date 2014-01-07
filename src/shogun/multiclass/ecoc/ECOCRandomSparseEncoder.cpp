/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <vector>
#include <limits>
#include <algorithm>

#include <multiclass/ecoc/ECOCRandomSparseEncoder.h>
#include <multiclass/ecoc/ECOCUtil.h>

using namespace shogun;

CECOCRandomSparseEncoder::CECOCRandomSparseEncoder(int32_t maxiter, int32_t codelen,
        float64_t pzero, float64_t pposone, float64_t pnegone)
    :m_maxiter(maxiter), m_codelen(codelen), m_pzero(pzero), m_pposone(pposone), m_pnegone(pnegone)
{
    if (!check_probability(pzero, pposone, pnegone))
        SG_ERROR("probability of 0, +1 and -1 must sum to one")

    init();
}

void CECOCRandomSparseEncoder::init()
{
    SG_ADD(&m_maxiter, "maxiter", "max number of iterations", MS_NOT_AVAILABLE);
    SG_ADD(&m_codelen, "codelen", "code length", MS_NOT_AVAILABLE);
    SG_ADD(&m_pzero, "pzero", "probability of 0", MS_NOT_AVAILABLE);
    SG_ADD(&m_pposone, "pposone", "probability of +1", MS_NOT_AVAILABLE);
    SG_ADD(&m_pnegone, "pnegone", "probability of -1", MS_NOT_AVAILABLE);
}

void CECOCRandomSparseEncoder::set_probability(float64_t pzero, float64_t pposone, float64_t pnegone)
{
    if (!check_probability(pzero, pposone, pnegone))
        SG_ERROR("probability of 0, +1 and -1 must sum to one")

    m_pzero   = pzero;
    m_pposone = pposone;
    m_pnegone = pnegone;
}

SGMatrix<int32_t> CECOCRandomSparseEncoder::create_codebook(int32_t num_classes)
{
    int32_t codelen = m_codelen;
    if (codelen <= 0)
        codelen = get_default_code_length(num_classes);


    SGMatrix<int32_t> best_codebook(codelen, num_classes, true);
    int32_t best_dist = 0;

    SGMatrix<int32_t> codebook(codelen, num_classes);
    std::vector<int32_t> random_sel(num_classes);
    int32_t n_iter = 0;

    while (true)
    {
        // fill codebook
        codebook.zero();
        for (int32_t i=0; i < codelen; ++i)
        {
            // randomly select two positions
            for (int32_t j=0; j < num_classes; ++j)
                random_sel[j] = j;
            std::random_shuffle(random_sel.begin(), random_sel.end());
            if (CMath::random(0.0, 1.0) > 0.5)
            {
                codebook(i, random_sel[0]) = +1;
                codebook(i, random_sel[1]) = -1;
            }
            else
            {
                codebook(i, random_sel[0]) = -1;
                codebook(i, random_sel[1]) = +1;
            }

            // assign the remaining positions
            for (int32_t j=2; j < num_classes; ++j)
            {
                float64_t randval = CMath::random(0.0, 1.0);
                if (randval > m_pzero)
                {
                    if (randval > m_pzero+m_pposone)
                        codebook(i, random_sel[j]) = -1;
                    else
                        codebook(i, random_sel[j]) = +1;
                }
            }
        }

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

        if (++n_iter >= m_maxiter)
            break;
    }

    return best_codebook;
}
