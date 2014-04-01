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

#include <shogun/lib/config.h>

#include <shogun/multiclass/ecoc/ECOCEncoder.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/Labels.h>

namespace shogun
{

/** Discriminant ECOC Encoder.
 *
 * A data-dependent ECOC coding scheme that learns a tree-style codebook. See the
 * following paper for details
 *
 *   Oriol Pujol, Petia Radeva, Jordi Vitria.  Discriminant ECOC: A Heuristic Method for
 *   Application Dependent Design of Error Correcting Output Codes. TPAMI 2006.
 *
 */
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

    /** set sffs iterations
     * @param iterations number of sffs iterations
     */
    void set_sffs_iterations(int32_t iterations) { m_iterations = iterations; }

    /** get sffs iterations
     */
    int32_t get_sffs_iterations() const { return m_iterations; }

    /** get name */
    virtual const char* get_name() const { return "ECOCDiscriminantEncoder"; }

    /** init codebook.
     * @param num_classes number of classes in this problem
     */
    virtual SGMatrix<int32_t> create_codebook(int32_t num_classes);

protected:
	/** init parameters */
    void init();

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    void binary_partition(const std::vector<int32_t>& classes);
    void run_sffs(std::vector<int32_t>& part1, std::vector<int32_t>& part2);
    float64_t sffs_iteration(float64_t MI, std::vector<int32_t>& part1, std::set<int32_t>& idata1,
            std::vector<int32_t>& part2, std::set<int32_t>& idata2);
    float64_t compute_MI(const std::set<int32_t>& idata1, const std::set<int32_t>& idata2);
    void compute_hist(int32_t i, float64_t max_val, float64_t min_val,
            const std::set<int32_t>& idata, int32_t *hist);

    int32_t m_iterations;
    int32_t m_num_trees;

    SGMatrix<int32_t> m_codebook;
    int32_t m_code_idx;
    CLabels *m_labels;
    CDenseFeatures<float64_t> *m_features;
    SGMatrix<float64_t> m_feats;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
};

} /* shogun */

#endif /* end of include guard: ECOCDISCRIMINANTENCODER_H__ */

