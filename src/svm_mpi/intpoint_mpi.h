/* -*-C++-*- */

#ifndef __INTPOINT_MPI_H_
#define __INTPOINT_MPI_H_

#include <sys/time.h>
#include "config.h"
#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
#include "mpi.h"
#endif

typedef struct {
  struct timeval distrib_utime;
  struct timeval distrib_stime;

  struct timeval add_pat_utime;
  struct timeval add_pat_stime;

  struct timeval overall_utime;
  struct timeval overall_stime;
  long overall_maxrss;
  unsigned long overall_max_matrix_bytes;
} IntpointResources;

#define REQ_M_PRIME 0
#define REQ_MAX_N 1
#define REQ_D 2
#define REQ_PATTERNS 3
#define REQ_MAX_KII_DIFF 4
#define REQ_NODE_MAX_KII_DIFF 5
#define REQ_M_FULL 6
#define REQ_SCALE_Z_COLS 7
#define REQ_CALC_Z_ZT 100
#define REQ_CALC_ZT_M 101
#define REQ_CALC_Z_M 102
#define REQ_CALC_ZT_Z_SCALED 103
#define REQ_CALC_NORM_Z 104
#define REQ_GATHER_RESOURCES 200
#define REQ_CHECKPOINT_DISTRIB 201
#define REQ_CHECKPOINT_ADD_PATTERNS 202
#define REQ_GATHER_KII 800  /* debugging only */
#define REQ_GATHER_KMNZ 801 /* debugging only */
#define REQ_MATRIX_MAX_MEMUSAGE 998
#define REQ_MATRIX_MEMUSAGE 999
#define REQ_QUIT 1000
#define REQ_READ_EXT_Z 1100
#define REQ_Z_COLUMNS 1101

/* Some helper functions */
void bcast_int(MPI_Comm comm, int root, int val);
void send_int(MPI_Comm comm, int dest, int tag, int val);
void bcast_req(MPI_Comm comm, int root, int req);

/* Resource helpers */
void resources_checkpoint_distrib_time(IntpointResources *res);
void resources_checkpoint_add_patterns_time(IntpointResources *res);
void resources_checkpoint_overall(IntpointResources *res);
void resources_print(IntpointResources *res, const unsigned num_nodes);

unsigned distribute_dimensions(const unsigned m_full,
			       const unsigned num_nodes,
			       const unsigned maxn, const unsigned d,
			       unsigned *m_last);

#endif /* ! __INTPOINT_MPI_H_ */
