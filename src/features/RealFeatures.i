%module RealFeatures

%{
    #include "features/RealFeatures.h" 
%}

%include "features/SimpleFeatures.i"
%include "lib/common.i"
%include "features/CharFeatures.i"

%feature("notabstract") CRealFeatures;
%include "carrays.i"

%array_class(double, doubleArray);

class CRealFeatures: public CSimpleFeatures<DREAL>
{
 public:
  CRealFeatures(LONG size) : CSimpleFeatures<DREAL>(size)
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

  bool Align_char_features(CCharFeatures* cf, CCharFeatures* Ref, DREAL gapCost) ;

  virtual CFeatures* duplicate() const;
  virtual EFeatureType get_feature_type() { return F_DREAL; }

  virtual bool load(CHAR* fname);
  virtual bool save(CHAR* fname);
 protected:
  DREAL Align(CHAR * seq1, CHAR* seq2, INT l1, INT l2, DREAL GapCost) ;

};

%pythoncode 
%{

def createDoubleArray(list):
   array = doubleArray(len(list))
   for i in range(len(list)):
      array[i] = list[i]
   return array

def createDoubleArray2(list):
   array = doubleArray2(len(list))
   for i in range(len(list)):
      array[i] = list[i]
   return array


%}

