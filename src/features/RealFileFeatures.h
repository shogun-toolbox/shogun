#ifndef _REALFILEFEATURES__H__
#define _REALFILEFEATURES__H__

#include "lib/common.h"
#include "features/RealFeatures.h"

class CRealFileFeatures: public CRealFeatures
{
 public:
  CRealFileFeatures(long size, FILE* file);
  CRealFileFeatures(long size, char* filename);

  CRealFileFeatures(const CRealFileFeatures& orig);

  virtual ~CRealFileFeatures();
  
  virtual REAL* load_feature_matrix();
  virtual CFeatures* duplicate() const;

  int get_label(long idx);

protected:
  /// compute feature vector for sample num
  /// len is returned by reference
  virtual REAL* compute_feature_vector(long num, long& len, REAL* target=NULL);

  bool load_base_data();

  FILE* working_file;
  char* working_filename;
  bool status;
  int* labels;

  unsigned char intlen;
  unsigned char doublelen;
  unsigned int endian;
  unsigned int fourcc;
  unsigned int preprocd;
  long filepos;
};
#endif
