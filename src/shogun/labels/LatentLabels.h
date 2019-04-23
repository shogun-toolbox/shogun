/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Evgeniy Andreev, Yuyu Zhang, Fernando Iglesias,
 *          Sergey Lisitsyn, Soeren Sonnenburg
 */

#ifndef __LATENTLABELS_H__
#define __LATENTLABELS_H__

#include <shogun/lib/config.h>

#include <shogun/labels/Labels.h>
#include <shogun/lib/Data.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
	/** @brief abstract class for latent labels
	 * As latent labels always depends on the given application, this class
	 * only defines the API that the user has to implement for latent labels.
	 */
	class LatentLabels : public Labels
	{
		public:
			/** default ctor */
			LatentLabels();

			/** constructor
			 *
			 * @param num_samples the number of labels
			 */
			LatentLabels(int32_t num_samples);

			/** constructor
			 *
			 * @param labels the (y_i) labels
			 */
			LatentLabels(std::shared_ptr<Labels> labels);

			/** destructor */
			virtual ~LatentLabels() override;

			/** get all the stored latent labels
			 *
			 * @return the DynamicObjectArray with the latent labels in it
			 */
			std::shared_ptr<DynamicObjectArray> get_latent_labels() const;

			/** get the latent label of a given example
			 *
			 * @param idx index of the label
			 * @return the user defined latent label
			 */
			std::shared_ptr<Data> get_latent_label(int32_t idx);

			/** append the latent label
			 *
			 * @param label latent label
			 */
			void add_latent_label(std::shared_ptr<Data> label);

			/** set latent label at a given index
			 *
			 * @param idx position of the label
			 * @param label the latent label
			 * @return TRUE if success, FALSE otherwise
			 */
			bool set_latent_label(int32_t idx, std::shared_ptr<Data> label);

		    virtual bool is_valid() const override;

		    /** Make sure the label is valid, otherwise raise SG_ERROR.
			 *
			 * possible with subset
			 *
			 * @param context optional message to convey the context
			 */
			virtual void ensure_valid(const char* context=NULL) override;

			/** get label type
			 *
			 * @return label type (binary, multiclass, ...)
			 */
			virtual ELabelType get_label_type() const override { return LT_LATENT; }

			/** Returns the name of the SGSerializable instance.
			 *
			 * @return name of the SGSerializable
			 */
			virtual const char* get_name() const override { return "LatentLabels"; }

			/** get the number of stored labels
			 *
			 * @return the number of labels
			 */
			virtual int32_t get_num_labels() const override;

			/** set labels
			 *
			 * @param labels the labels (y_i)
			 */
			void set_labels(std::shared_ptr<Labels> labels);

			/** get the labels (y_i)
			 *
			 * @return the labels (y_i)
			 */
			std::shared_ptr<Labels> get_labels() const;

		protected:
			/** the of Data, the latent labels (h_i) */
			std::shared_ptr<DynamicObjectArray> m_latent_labels;
			/** the labels (y_i) */
			std::shared_ptr<Labels> m_labels;

		private:
			/** initalize the values to default values */
			void init();
	};
}

#endif /* __LATENTLABELS_H__ */

