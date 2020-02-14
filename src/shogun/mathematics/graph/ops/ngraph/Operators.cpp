#include <shogun/base/manifest.h>
#include <shogun/mathematics/graph/ops/ngraph/Add.h>

using namespace shogun;

BEGIN_OPERATOR_MANIFEST("Shogun's graph library")
EXPORT_OPERATOR(AddNGraph, OperatorNGraphBackend, "Add")
END_OPERATOR_MANIFEST()