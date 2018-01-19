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
class CECOCStrategy: public CMulticlassStrategy
{
public:
    /** default constructor, do not call, only to make serializer happy */
    CECOCStrategy();

    /** constructor */
    CECOCStrategy(CECOCEncoder *encoder, CECOCDecoder *decoder);

    /** destructor */
    virtual ~CECOCStrategy();

    /** get name */
    virtual const char* get_name() const
    {
        return "ECOCStrategy";
    }

    /** start training */
    virtual void train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels);

    /** has more training phase */
    virtual bool train_has_more();

    /** prepare for the next training phase.
     * @return The subset that should be applied. Return NULL when no subset is needed.
     */
    virtual SGVector<int32_t> train_prepare_next();

    /** decide the final label.
     * @param outputs a vector of output from each machine (in that order)
     */
    virtual int32_t decide_label(SGVector<float64_t> outputs);

    /** get number of machines used in this strategy.
     */
    virtual int32_t get_num_machines();

protected:
    /** ECOC encoder */
    CECOCEncoder *m_encoder;
    /** ECOC decoder */
    CECOCDecoder *m_decoder;

    /** ECOC codebook */
    SGMatrix<int32_t> m_codebook;

private:
	/** init parameters */
    void init();
};


}

#endif /* end of include guard: ECOCSTRATEGY_H__ */
