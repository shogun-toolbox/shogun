/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 *** Authors:                                                               ***
 ***  Thomas Serafini, Luca Zanni                                           ***
 ***   Dept. of Mathematics, University of Modena and Reggio Emilia - ITALY ***
 ***   serafini.thomas@unimo.it, zanni.luca@unimo.it                        ***
 ***  Gaetano Zanghirati                                                    ***
 ***   Dept. of Mathematics, University of Ferrara - ITALY                  ***
 ***   g.zanghirati@unife.it                                                ***
 ***                                                                        ***
 *** Software homepage: http://dm.unife.it/gpdt                             ***
 ***                                                                        ***
 *** This work is supported by the Italian FIRB Projects                    ***
 ***      'Statistical Learning: Theory, Algorithms and Applications'       ***
 ***      (grant RBAU01877P), http://slipguru.disi.unige.it/ASTA            ***
 *** and                                                                    ***
 ***      'Parallel Algorithms and Numerical Nonlinear Optimization'        ***
 ***      (grant RBAU01JYPN), http://dm.unife.it/pn2o                       ***
 ***                                                                        ***
 *** Copyright (C) 2004 by T. Serafini, G. Zanghirati, L. Zanni.            ***
 */

#include "base/SGObject.h"

class QPproblem : public CSGObject
{
// ----------------- Public Data ---------------
public:
  int     chunk_size;
  int     ell;
  int    *y;
  double DELTAsv;
  int     q;
  int     maxmw;
  double  c_const;
  double  bee;
  double  delta;
  bool linadd;

  sKernel* KER;
  int     ker_type;
  int     projection_solver, projection_projector; 
  int     PreprocessMode;
  int     preprocess_size;
  int     verbosity;
  double  tau_proximal;
  double objective_value;

// ----------------- Public Methods ---------------
  QPproblem ();
  ~QPproblem();
  int  ReadSVMFile    (char *fInput);
  int  ReadGPDTBinary(char *fName);
  int  Check2Class    (void);
  void Subproblem     (QPproblem &ker, int len, int *perm);
  void PrepMP         (void);

  double  gpdtsolve      (double *solution);
  double  pgpdtsolve     (double *solution);
  inline bool get_linadd_enabled()
  {
	  return linadd;
  }

// ----------------- Private Data  ---------------
private:
  int    dim;
  int    *index_in, *index_out;
  int    *ing, *inaux, *inold, *incom;
  int    *cec;
  int    nb;
  int    *bmem, *bmemrid, *pbmr;
  int    my_chunk_size;    // chunk_size for the current processor
  int    my_spD_offset;    // offset of the current processor into sp_D matrix
  int    recvl[32], displ[32];
  double kktold;
  double DELTAvpm, InitialDELTAvpm, DELTAkin;
  double *alpha;
  double *grad, *st;

// ----------------- Private Methods ---------------
private:
  int  Preprocess0 (int *aux, int *sv);
  int  Preprocess1 (sKernel* KER, int *aux, int *sv);
  int  optimal     (void);

  bool is_zero(int  i) { return (alpha[i] < DELTAsv); }
  bool is_free(int  i) 
       { return (alpha[i] > DELTAsv && alpha[i] < (c_const - DELTAsv)); }
  bool is_bound(int i) { return (alpha[i] > (c_const - DELTAsv)); }

};
