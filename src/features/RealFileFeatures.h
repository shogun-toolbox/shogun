#ifndef _REALFILEFEATURES__H__
#define _REALFILEFEATURES__H__

#include "lib/common.h"
#include "features/RealFeatures.h"

class CRealFileFeatures: public CRealFeatures
{
 public:
  CRealFileFeatures(FILE* file);
  CRealFileFeatures(char* filename);

  CRealFileFeatures(const CRealFileFeatures& orig);

  virtual ~CRealFileFeatures();
  
  /** set feature matrix
      necessary to set feature_matrix, num_features, num_vectors, where
      num_features is the column offset, and columns are linear in memory
      see below for definition of feature_matrix
  */
  virtual REAL* set_feature_matrix();
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
