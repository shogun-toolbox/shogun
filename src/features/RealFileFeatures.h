#ifndef _REALFILEFEATURES__H__
#define _REALFILEFEATURES__H__

#include "lib/common.h"
#include "features/RealFeatures.h"

class CRealFileFeatures: public CRealFeatures
{
 public:
  CRealFileFeatures(LONG size, FILE* file);
  CRealFileFeatures(LONG size, CHAR* filename);

  CRealFileFeatures(const CRealFileFeatures& orig);

  virtual ~CRealFileFeatures();
  
  virtual REAL* load_feature_matrix();
  virtual CFeatures* duplicate() const;

  INT get_label(INT idx);

protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual REAL* compute_feature_vector(INT num, INT& len, REAL* target=NULL);

  bool load_base_data();

  FILE* working_file;
  CHAR* working_filename;
  bool status;
  INT* labels;

  BYTE intlen;
  BYTE doublelen;
  UINT endian;
  UINT fourcc;
  UINT preprocd;
  LONG filepos;
};
#endif
