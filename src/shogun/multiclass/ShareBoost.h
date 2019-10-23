/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Yuyu Zhang, Bjoern Esser
 */

#ifndef SHAREBOOST_H__
#define SHAREBOOST_H__

#include <shogun/lib/config.h>

#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>

namespace shogun
{

/** ShareBoost is a linear multiclass algorithm that efficiently
 * learns a subset of features shared by all classes.
 *
 * See the following paper for details:
 *
 *     Shai Shalev-Shwartz, Yonatan Wexler, Amnon Shashua. ShareBoost: Efficient
 *     Multiclass Learning with Feature Sharing. NIPS 2011.
 */
class ShareBoost: public LinearMulticlassMachine
{
public:
    /** default constructor */
	ShareBoost();

	/** constructor */
	ShareBoost(const std::shared_ptr<DenseFeatures<float64_t> >&features, const std::shared_ptr<MulticlassLabels >&labs, int32_t num_nonzero_feas);

    /** destructor */
	virtual ~ShareBoost() {}

    /** get name */
    virtual const char* get_name() const { return "ShareBoost"; }

	/** set number of non-zero features the algorithm should seek */
	void set_num_nonzero_feas(int32_t n) { m_nonzero_feas = n; }

	/** get number of non-zero features the algorithm should seek */
	int32_t get_num_nonzero_feas() const { return m_nonzero_feas; }

	/** assign features */
	void set_features(const std::shared_ptr<Features >&f);

	/** get active set */
	SGVector<int32_t> get_activeset();

	friend class ShareBoostOptimizer;
protected:

	/** train machine */
	virtual bool train_machine(std::shared_ptr<Features> data = NULL);

private:
	void init_sb_params(); ///< init machine parameters

	void compute_rho(); ///< compute the rho matrix
	int32_t choose_feature(); ///< choose next feature greedily
	void optimize_coefficients(); ///< optimize coefficients with gradient descent
	void compute_pred(); ///< compute predictions on training data, according to W in m_machines
	void compute_pred(const float64_t *W); ///< compute predictions on training data, according to given W

	int32_t m_nonzero_feas; ///< number of non-zero features to seek
	SGVector<int32_t> m_activeset; ///< selected features

	SGMatrix<float64_t> m_fea; ///< feature matrix used during training
	SGMatrix<float64_t> m_rho; ///< cache_matrix for rho
	SGVector<float64_t> m_rho_norm; ///< column sum of m_rho
	SGMatrix<float64_t> m_pred; ///< predictions, used in training
};

} /* shogun */

#endif /* end of include guard: SHAREBOOST_H__ */

