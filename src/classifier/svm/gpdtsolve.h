
class QPproblem
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
  void write_solution (FILE *fp, double *sol);

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
