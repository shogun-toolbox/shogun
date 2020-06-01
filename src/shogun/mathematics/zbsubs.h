/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef ZBSUBS_H
#define ZBSUBS_H

int zbesh_(
    double* zr, double* zi, double* fnu, int* kode, int* m, int* n, double* cyr,
    double* cyi, int* nz, int* ierr);
int zbesi_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* cyr,
    double* cyi, int* nz, int* ierr);
int zbesj_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* cyr,
    double* cyi, int* nz, int* ierr);
int zbesk_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* cyr,
    double* cyi, int* nz, int* ierr);
int zbesy_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* cyr,
    double* cyi, int* nz, double* cwrkr, double* cwrki, int* ierr);
int zairy_(
    double* zr, double* zi, int* id, int* kode, double* air, double* aii,
    int* nz, int* ierr);
int zbiry_(
    double* zr, double* zi, int* id, int* kode, double* bir, double* bii,
    int* ierr);
int zmlt_(
    double* ar, double* ai, double* br, double* bi, double* cr, double* ci);
int zdiv_(
    double* ar, double* ai, double* br, double* bi, double* cr, double* ci);
int zsqrt_(double* ar, double* ai, double* br, double* bi);
int zexp_(double* ar, double* ai, double* br, double* bi);
int zlog_(double* ar, double* ai, double* br, double* bi, int* ierr);
double zabs_(double* zr, double* zi);
int zbknu_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, double* tol, double* elim, double* alim);
int zkscl_(
    double* zrr, double* zri, double* fnu, int* n, double* yr, double* yi,
    int* nz, double* rzr, double* rzi, double* ascle, double* tol,
    double* elim);
int zshch_(
    double* zr, double* zi, double* cshr, double* cshi, double* cchr,
    double* cchi);
int zrati_(
    double* zr, double* zi, double* fnu, int* n, double* cyr, double* cyi,
    double* tol);
int zs1s2_(
    double* zrr, double* zri, double* s1r, double* s1i, double* s2r,
    double* s2i, int* nz, double* ascle, double* alim, int* iuf);
int zbunk_(
    double* zr, double* zi, double* fnu, int* kode, int* mr, int* n, double* yr,
    double* yi, int* nz, double* tol, double* elim, double* alim);
int zmlri_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, double* tol);
int zwrsk_(
    double* zrr, double* zri, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, double* cwr, double* cwi, double* tol, double* elim,
    double* alim);
int zseri_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, double* tol, double* elim, double* alim);
int zasyi_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, double* rl, double* tol, double* elim, double* alim);
int zuoik_(
    double* zr, double* zi, double* fnu, int* kode, int* ikflg, int* n,
    double* yr, double* yi, int* nuf, double* tol, double* elim, double* alim);
int zacon_(
    double* zr, double* zi, double* fnu, int* kode, int* mr, int* n, double* yr,
    double* yi, int* nz, double* rl, double* fnul, double* tol, double* elim,
    double* alim);
int zbinu_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* cyr,
    double* cyi, int* nz, double* rl, double* fnul, double* tol, double* elim,
    double* alim);
double dgamln_(double* z__, int* ierr);
int zacai_(
    double* zr, double* zi, double* fnu, int* kode, int* mr, int* n, double* yr,
    double* yi, int* nz, double* rl, double* tol, double* elim, double* alim);
int zuchk_(double* yr, double* yi, int* nz, double* ascle, double* tol);
int zunik_(
    double* zrr, double* zri, double* fnu, int* ikflg, int* ipmtr, double* tol,
    int* init, double* phir, double* phii, double* zeta1r, double* zeta1i,
    double* zeta2r, double* zeta2i, double* sumr, double* sumi, double* cwrkr,
    double* cwrki);
int zunhj_(
    double* zr, double* zi, double* fnu, int* ipmtr, double* tol, double* phir,
    double* phii, double* argr, double* argi, double* zeta1r, double* zeta1i,
    double* zeta2r, double* zeta2i, double* asumr, double* asumi, double* bsumr,
    double* bsumi);
int zunk1_(
    double* zr, double* zi, double* fnu, int* kode, int* mr, int* n, double* yr,
    double* yi, int* nz, double* tol, double* elim, double* alim);
int zunk2_(
    double* zr, double* zi, double* fnu, int* kode, int* mr, int* n, double* yr,
    double* yi, int* nz, double* tol, double* elim, double* alim);
int zbuni_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, int* nui, int* nlast, double* fnul, double* tol,
    double* elim, double* alim);
int zuni1_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, int* nlast, double* fnul, double* tol, double* elim,
    double* alim);
int zuni2_(
    double* zr, double* zi, double* fnu, int* kode, int* n, double* yr,
    double* yi, int* nz, int* nlast, double* fnul, double* tol, double* elim,
    double* alim);

#endif // ZBSUBS_H
