
double ddot_(int len, double*a, int i1, double*b, int i2) 
{
  double sum=0 ;
  int i ;
  for (i=0; i<len; i++)
    sum+=a[i]*b[i] ;
  return sum ;
} ;
