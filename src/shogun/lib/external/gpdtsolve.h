/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
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
 ***                                                                        ***
 *** SHOGUN adaptions  Written (W) 2006-2008 Soeren Sonnenburg              ***
 */

#include <base/SGObject.h>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace shogun
{
/** class QProblem */
class QPproblem
{
// ----------------- Public Data ---------------
public:
  /** chunk size */
  int32_t     chunk_size;
  /** ell */
  int32_t     ell;
  /** y */
  int32_t    *y;
  /** delta sv */
  float64_t DELTAsv;
  /** q */
  int32_t     q;
  /** max mw */
  int32_t     maxmw;
  /** c const */
  float64_t  c_const;
  /** bee */
  float64_t  bee;
  /** delta */
  float64_t  delta;
  /** linadd */
  bool linadd;

  /** kernel */
  sKernel* KER;
  /** kernel type */
  int32_t     ker_type;
  /** projection solver */
  int32_t     projection_solver;
  /** projection projector */
  int32_t     projection_projector;
  /** preprocess mode */
  int32_t     PreprocessMode;
  /** preprocess size */
  int32_t     preprocess_size;
  /** verbosity */
  int32_t     verbosity;
  /** tau proximal */
  float64_t  tau_proximal;
  /** objective value */
  float64_t objective_value;

// ----------------- Public Methods ---------------
  /** constructor */
  QPproblem ();
  ~QPproblem();

  /** read SVM file
   *
   * @param fInput input filename
   * @return an int
   */
  int32_t  ReadSVMFile    (char *fInput);

  /** read GPDT binary
   *
   * @param fName input filename
   * @return an int
   */
  int32_t  ReadGPDTBinary(char *fName);

  /** check if 2-class
   *
   * @return an int
   */
  int32_t  Check2Class    ();

  /** subproblem
   *
   * @param dst where to copy subproblem to
   * @param ker problem kernel
   * @param len length
   * @param perm perm
   */
  static void copy_subproblem(QPproblem* dst, QPproblem* ker, int32_t len, int32_t *perm);

  /** PrepMP */
  void PrepMP         ();

  /** solve gpdt
   *
   * @param solution
   * @return something floaty
   */
  float64_t  gpdtsolve      (float64_t *solution);

  /** solve pgpdt
   *
   * @param solution
   * @return something floaty
   */
  float64_t  pgpdtsolve     (float64_t *solution);

  /** check if lineadd is enabled
   *
   * @return if lineadd is enabled
   */
  inline bool get_linadd_enabled()
  {
    return linadd;
  }

  /** @return object name */
  virtual const char* get_name() const { return "QPproblem"; }

// ----------------- Private Data  ---------------
private:
  int32_t    dim;
  int32_t    *index_in, *index_out;
  int32_t    *ing, *inaux, *inold, *incom;
  int32_t    *cec;
  int32_t    nb;
  int32_t    *bmem, *bmemrid, *pbmr;
  int32_t    my_chunk_size;  // chunk_size for the current processor
  int32_t    my_spD_offset;  // offset of the current processor into sp_D matrix
  int32_t    recvl[32], displ[32];
  float64_t kktold;
  float64_t DELTAvpm, InitialDELTAvpm, DELTAkin;
  float64_t *alpha;
  float64_t *grad, *st;

// ----------------- Private Methods ---------------
private:
  int32_t  Preprocess0 (int32_t *aux, int32_t *sv);
  int32_t  Preprocess1 (sKernel* KER, int32_t *aux, int32_t *sv);
  int32_t  optimal     ();

  bool is_zero(int32_t  i) { return (alpha[i] < DELTAsv); }
  bool is_free(int32_t  i)
       { return (alpha[i] > DELTAsv && alpha[i] < (c_const - DELTAsv)); }
  bool is_bound(int32_t i) { return (alpha[i] > (c_const - DELTAsv)); }

};
}
#endif // DOXYGEN_SHOULD_SKIP_THIS
