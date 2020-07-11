/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Soeren Sonnenburg, Yuyu Zhang
 */

#ifndef ECOCSTRATEGY_H__
#define ECOCSTRATEGY_H__

#include <shogun/lib/config.h>

#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/multiclass/ecoc/ECOCEncoder.h>
#include <shogun/multiclass/ecoc/ECOCDecoder.h>

namespace shogun
{

/** Multiclass Strategy that uses ECOC coding */
class ECOCStrategy: public MulticlassStrategy
{
public:
    /** default constructor, do not call, only to make serializer happy */
    ECOCStrategy();

    /** constructor */
    ECOCStrategy(std::shared_ptr<ECOCEncoder >encoder, std::shared_ptr<ECOCDecoder >decoder);

    /** destructor */
    ~ECOCStrategy() override;

    /** get name */
    const char* get_name() const override
    {
        return "ECOCStrategy";
    }

    /** start training */
    void train_start(std::shared_ptr<MulticlassLabels >orig_labels, std::shared_ptr<BinaryLabels >train_labels) override;

    /** has more training phase */
    bool train_has_more() override;

    /** prepare for the next training phase.
     * @return The subset that should be applied. Return NULL when no subset is needed.
     */
    SGVector<int32_t> train_prepare_next() override;

    /** decide the final label.
     * @param outputs a vector of output from each machine (in that order)
     */
    int32_t decide_label(SGVector<float64_t> outputs) override;

    /** get number of machines used in this strategy.
     */
    int32_t get_num_machines() override;

protected:
    /** ECOC encoder */
    std::shared_ptr<ECOCEncoder >m_encoder;
    /** ECOC decoder */
    std::shared_ptr<ECOCDecoder >m_decoder;

    /** ECOC codebook */
    SGMatrix<int32_t> m_codebook;

private:
	/** init parameters */
    void init();
};


}

#endif /* end of include guard: ECOCSTRATEGY_H__ */
