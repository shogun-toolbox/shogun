/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef ECOCDISCRIMINANTENCODER_H__
#define ECOCDISCRIMINANTENCODER_H__

#include <vector>
#include <set>

#include <shogun/multiclass/ecoc/ECOCEncoder.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Labels.h>

namespace shogun
{

class CECOCDiscriminantEncoder: public CECOCEncoder
{
public:
    /** constructor */
    CECOCDiscriminantEncoder();

    /** destructor */
    virtual ~CECOCDiscriminantEncoder();

    /** set features */
    void set_features(CDenseFeatures<float64_t> *features);

    /** set labels */
    void set_labels(CLabels *labels);

    /** get name */
    virtual const char* get_name() const { return "ECOCDiscriminantEncoder"; }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);

private:
    void init();


    void binary_partition(const std::vector<int32_t>& classes);
    void run_sffs(std::vector<int32_t>& part1, std::vector<int32_t>& part2);
    float64_t sffs_iteration(float64_t MI, std::vector<int32_t>& part1, std::set<int32_t>& idata1,
            std::vector<int32_t>& part2, std::set<int32_t>& idata2);
    float64_t compute_MI(const std::set<int32_t>& idata1, const std::set<int32_t>& idata2);
    void compute_hist(int32_t i, float64_t max_val, float64_t min_val, 
            const std::set<int32_t>& idata, int32_t *hist);

    int32_t m_iterations;

    SGMatrix<int32_t> m_codebook;
    int32_t m_code_idx;
    CLabels *m_labels;
    CDenseFeatures<float64_t> *m_features;
    SGMatrix<float64_t> m_feats;
};

} /* shogun */ 

#endif /* end of include guard: ECOCDISCRIMINANTENCODER_H__ */

