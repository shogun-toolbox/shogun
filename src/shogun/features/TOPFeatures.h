/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _CTOPFEATURES__H__
#define _CTOPFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/HMM.h>

namespace shogun
{
template <class T> class DenseFeatures;
class HMM;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** HMM indices */
struct T_HMM_INDIZES
{
	/** index p */
	int32_t* idx_p;
	/** index q */
	int32_t* idx_q;
	/** index a rows */
	int32_t* idx_a_rows;
	/** index a cols */
	int32_t* idx_a_cols;
	/** index b rows */
	int32_t* idx_b_rows;
	/** index b cols */
	int32_t* idx_b_cols;

	/** number p */
	int32_t num_p;
	/** number q */
	int32_t num_q;
	/** number a */
	int32_t num_a;
	/** number b */
	int32_t num_b;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief The class TOPFeatures implements TOP kernel features obtained from
 * two Hidden Markov models.
 *
 * It was used in
 *
 * K. Tsuda, M. Kawanabe, G. Raetsch, S. Sonnenburg, and K.R. Mueller. A new
 * discriminative kernel from probabilistic models. Neural Computation,
 * 14:2397-2414, 2002.
 *
 * which also has the details.
 *
 * Note that TOP-features are computed on the fly, so to be effective feature
 * caching should be enabled.
 *
 * It inherits its functionality from DenseFeatures, which should be
 * consulted for further reference.
 */
class TOPFeatures : public DenseFeatures<float64_t>
{
	public:
		/** default constructor  */
		TOPFeatures();

		/** constructor
		 *
		 * @param size cache size
		 * @param p positive HMM
		 * @param n negative HMM
		 * @param neglin if negative HMM is of linear shape
		 * @param poslin if positive HMM is of linear shape
		 */
		TOPFeatures(int32_t size, std::shared_ptr<HMM> p, std::shared_ptr<HMM> n, bool neglin, bool poslin);

		/** copy constructor */
		TOPFeatures(const TOPFeatures &orig);

		virtual ~TOPFeatures();

		/** set HMMs
		 *
		 * @param p positive HMM
		 * @param n negative HMM
		 */
		void set_models(const std::shared_ptr<HMM>& p, const std::shared_ptr<HMM>& n);

		/** set feature matrix
		 *
		 * @return something floaty
		 */
		virtual float64_t* set_feature_matrix();

		/** compute number of features
		 *
		 * @return number of features
		 */
		int32_t compute_num_features();

		/** compute relevant indices
		 *
		 * @param hmm HMM to compute for
		 * @param hmm_idx HMM index
		 * @return if computing was successful
		 */
		bool compute_relevant_indizes(const std::shared_ptr<HMM>& hmm, T_HMM_INDIZES* hmm_idx);

		/** @return object name */
		virtual const char* get_name() const { return "TOPFeatures"; }

	protected:
		/** compute feature vector
		 *
		 * @param num num
		 * @param len len
		 * @param target
		 * @return something floaty
		 */
		virtual float64_t* compute_feature_vector(
			int32_t num, int32_t& len, float64_t* target=NULL) const;

		/** computes the feature vector to the address addr
		 *
		 * @param addr address
		 * @param num num
		 * @param len len
		 */
		void compute_feature_vector(float64_t* addr, int32_t num, int32_t& len) const;

	private:
		void init();

	protected:
		/** positive HMM */
		std::shared_ptr<HMM> pos;
		/** negative HMM */
		std::shared_ptr<HMM> neg;
		/** if negative HMM is a LinearHMM */
		bool neglinear;
		/** if positive HMM is a LinearHMM */
		bool poslinear;

		/** positive relevant indices */
		T_HMM_INDIZES pos_relevant_indizes;
		/** negative relevant indices */
		T_HMM_INDIZES neg_relevant_indizes;
};
}
#endif
