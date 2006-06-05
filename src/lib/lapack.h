#ifndef _LAPACK_H__
#define _LAPACK_H__

#include "lib/common.h"

extern "C" {

INT dsyev_(CHAR*, CHAR*, int*, double*, int*, double*, double*, int*, int*);

}
#endif
