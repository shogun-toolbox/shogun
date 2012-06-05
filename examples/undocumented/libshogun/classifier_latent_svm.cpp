#include <shogun/features/LatentLabels.h>
#include <shogun/features/LatentFeatures.h>
#include <shogun/classifier/svm/LatentLinearMachine.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

int main (int argc, char** argv)
{
  init_shogun_with_defaults ();

  /* read the features/labels */

  /* train the classifier */
  float64_t C = 10.0;
  CLatentFeatures* latent_features = NULL;
  CLatentLabels* latent_labels = NULL;
  //CLatentLinearMachine llm (C, latent_features, latent_labels);
  CLatentLinearMachine llm;



  exit_shogun ();
  return 0;
}
