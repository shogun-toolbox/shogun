#include <mathematics/Math.h>
#include <io/SGIO.h>

namespace shogun
{
#define OCAS_PLUS_INF CMath::INFTY
#define OCAS_CALLOC(...) calloc(__VA_ARGS__)
#define OCAS_FREE(...) SG_FREE(__VA_ARGS__)

#define INDEX2(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
}
