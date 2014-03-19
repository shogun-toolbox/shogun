#ifndef OPTDEFINES_H
#define OPTDEFINES_H
#include <shogun/lib/common.h>
#include <shogun/structure/BmrmStatistics.h>
#include <shogun/io/SGIO.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/lib/external/libqp.h>
#include <shogun/lib/Time.h>
#include <climits>
#include <limits>

#define BMRM_PLUS_INF (-log(0.0))
#define BMRM_CALLOC(x, y) SG_CALLOC(y, x)
#define BMRM_REALLOC(x, y) SG_REALLOC(x, y)
#define BMRM_FREE(x) SG_FREE(x)
#define BMRM_MEMCPY(x, y, z) memcpy(x, y, z)
#define BMRM_MEMMOVE(x, y, z) memmove(x, y, z)
#define BMRM_INDEX(ROW, COL, NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define BMRM_ABS(A) ((A) < 0 ? -(A) : (A))

#endif
