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

  void set_hmms(CHMM *pos, CHMM* neg) 
    {
      top_features.set_models(pos,neg) ;
    } ;
 protected:
  CGUI* gui ;
  CFeatures *train_features ;
  CTOPFeatures top_features ;

};
#endif
