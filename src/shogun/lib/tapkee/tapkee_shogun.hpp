#ifndef TAPKEE_SHOGUN_ADAPTER
#define TAPKEE_SHOGUN_ADAPTER

#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

namespace shogun
{

CDenseFeatures<float64_t>* tapkee_embed(int32_t N, CKernel* kernel, CDistance* distance, CDotFeatures* features);
}

#endif 
