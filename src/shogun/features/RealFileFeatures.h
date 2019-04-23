/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Sergey Lisitsyn
 */

#ifndef _DREALFILEFEATURES__H__
#define _DREALFILEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief The class RealFileFeatures implements a dense double-precision floating
 * point matrix <b>from a file</b>.
 *
 * It inherits its functionality from DenseFeatures, which should be
 * consulted for further reference.
 */
class RealFileFeatures: public DenseFeatures<float64_t>
{
	public:
		/** default constructor */
		RealFileFeatures();

		/** constructor
		 *
		 * @param size cache size
		 * @param file file to load features from
		 */
		RealFileFeatures(int32_t size, FILE* file);

		/** constructor
		 *
		 * @param size cache size
		 * @param filename filename to load features from
		 */
		RealFileFeatures(int32_t size, char* filename);

		/** copy constructor */
		RealFileFeatures(const RealFileFeatures& orig);

		virtual ~RealFileFeatures();

		/** load feature matrix
		 *
		 * @return loaded feature matrix
		 */
		virtual float64_t* load_feature_matrix();

		/** get label at given index
		 *
		 * @param idx index to look at
		 * @return label at given index
		 */
		int32_t get_label(int32_t idx);

		/** @return object name */
		virtual const char* get_name() const { return "RealFileFeatures"; }

	protected:
		/** compute feature vector for sample num
		 * len is returned by reference
		 *
		 * @param num num
		 * @param len len
		 * @param target target
		 */
		virtual float64_t* compute_feature_vector(
			int32_t num, int32_t& len, float64_t* target=NULL) const;

		/** load base data
		 *
		 * @return if loading was successful
		 */
		bool load_base_data();

	private:
		/** initialises members */
		void init();

    protected:
		/** working file */
		FILE* working_file;
		/** working filename */
		char* working_filename;
		/** status */
		bool status;
		/** labels */
		int32_t* labels;

		/** intlen */
		uint8_t intlen;
		/** doublelen */
		uint8_t doublelen;
		/** endian */
		uint32_t endian;
		/** fourcc */
		uint32_t fourcc;
		/** preprocd */
		uint32_t preprocd;
		/** filepos */
		int64_t filepos;
};
}
#endif
