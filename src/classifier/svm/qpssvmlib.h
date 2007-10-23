int qpssvm_solver(const void* (*get_col)(uint32_t),
                  double *diag_H,
                  double *f,
                  double b,
                  uint16_t *I,
                  double *x,
                  uint32_t n,
                  uint32_t tmax,
                  double tolabs,
                  double tolrel,
                  double *QP,
                  double *QD,
                  uint32_t verb);

