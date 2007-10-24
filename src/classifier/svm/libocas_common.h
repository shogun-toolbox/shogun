#include "lib/Mathematics.h"
#include "lib/io.h"

#define OCAS_PLUS_INF CMath::INFTY
#define OCAS_CALLOC(x...) calloc(x)
#define OCAS_FREE(x...) free(x)

#define INDEX2(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define ABS(A) ((A) < 0 ? -(A) : (A))
