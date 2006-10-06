%module RealFeatures

%{
    #include "features/RealFeatures.h" 
%}

%include "lib/common.i"
%include "features/SimpleFeatures.i"

%feature("notabstract") CRealFeatures;

class CRealFeatures: public CSimpleFeatures<DREAL>
{
 public:
  CRealFeatures(INT size) : CSimpleFeatures<DREAL>(size)
  {
  }

  CRealFeatures(const CRealFeatures & orig) : CSimpleFeatures<DREAL>(orig)
  {
  }

  CRealFeatures(DREAL* feature_matrix, INT num_feat, INT num_vec) : CSimpleFeatures<DREAL>(feature_matrix, num_feat, num_vec)
  {
  }

  CRealFeatures(CHAR* fname) : CSimpleFeatures<DREAL>(fname)
  {
    load(fname);
  }
};
