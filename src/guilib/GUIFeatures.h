#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "features/Features.h"
#include "features/TOPFeatures.h"

class CGUI ;

class CGUIFeatures
{
public:
  CGUIFeatures(CGUI *);
  ~CGUIFeatures();

  CFeatures *get_train_features() { return train_features ; } ;
  
 protected:
  CGUI* gui ;
  CFeatures *train_features ;
  CTOPFeatures top_features ;

};
#endif
