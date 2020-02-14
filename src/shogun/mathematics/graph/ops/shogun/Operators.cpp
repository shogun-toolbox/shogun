#include <shogun/base/manifest.h>
#include <shogun/mathematics/graph/ops/shogun/Add.h>
#include <shogun/mathematics/graph/ops/shogun/Input.h>

using namespace shogun;

BEGIN_OPERATOR_MANIFEST("Shogun's graph library")
EXPORT_OPERATOR(InputShogun, OperatorShogunBackend, "Input")
EXPORT_OPERATOR(AddShogun, OperatorShogunBackend, "Add")
END_OPERATOR_MANIFEST()