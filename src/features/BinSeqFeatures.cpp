#include <assert.h>
#include "features/BinSeqFeatures.h"

CBinSeqFeatures::CBinSeqFeatures(long size, CObservation *pos_, CObservation *neg_)
  : CShortFeatures(size), pos(pos_), neg(neg_)
{
  assert(pos!=NULL) ;
  assert(neg!=NULL) ;

  num_vectors = pos->get_DIMENSION() + neg->get_DIMENSION() ;

  if (pos->get_DIMENSION()>0)
    num_features=pos->get_obs_T(0) ;

  if ((pos->get_DIMENSION()>0) && (neg->get_DIMENSION()>0))
    assert(pos->get_obs_T(0)==neg->get_obs_T(0)) ;
} ;

CBinSeqFeatures::~CBinSeqFeatures()
{
  neg=pos=NULL ;
} ;

int CBinSeqFeatures::get_label(long idx) 
{
  if (idx<pos->get_DIMENSION())
    return 1;
  return -1 ;
} ;

long CBinSeqFeatures::get_number_of_examples() 
{
  return num_vectors ;
} ;

void CBinSeqFeatures::compute_feature_vector(long num, short int* feat)
{
  long num_pos=pos->get_DIMENSION() ;
  if (num<num_pos)
    {
      assert(pos!=NULL) ;
      assert(pos->get_obs_T(num)==num_features) ;
      for (long i=0; i<num_features; i++)
	feat[i]=pos->get_obs(num, i) ;
    } 
  else
    {
      assert(neg!=NULL) ;
      assert(neg->get_obs_T(num-num_pos)==num_features) ;
      for (long i=0; i<num_features; i++)
	feat[i]=neg->get_obs(num-num_pos, i) ;
    } ;
} ;

