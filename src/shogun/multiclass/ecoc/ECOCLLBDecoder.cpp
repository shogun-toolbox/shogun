#include <multiclass/ecoc/ECOCLLBDecoder.h>

using namespace shogun;

float64_t CECOCLLBDecoder::compute_distance(SGVector<float64_t> outputs, const int32_t *code)
{
    float64_t loss = 0;
    for (int32_t i=0; i < outputs.vlen; ++i)
        loss += outputs[i]*code[i];
    return -loss;
}
