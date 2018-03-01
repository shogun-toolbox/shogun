/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Viktor Gal, Fernando Iglesias,
 *          Wu Lin, Sergey Lisitsyn, Sanuj Sharma, Josh Klontz,
 *          Shashwat Lal Das, Philippe Tillet, Evan Shelhamer, Saurabh Goyal
 */
#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

#include <stdlib.h>

#ifndef NAN
#include <stdlib.h>
#define NAN (strtod("NAN",NULL))
#endif


using namespace shogun;

#ifdef USE_LOGCACHE
#ifdef USE_HMMDEBUG
#define MAX_LOG_TABLE_SIZE 10*1024*1024
#define LOG_TABLE_PRECISION 1e-6
#else //USE_HMMDEBUG
#define MAX_LOG_TABLE_SIZE 123*1024*1024
#define LOG_TABLE_PRECISION 1e-15
#endif //USE_HMMDEBUG
int32_t CMath::LOGACCURACY         = 0; // 100000 steps per integer
#endif // USE_LOGCACHE

int32_t CMath::LOGRANGE            = 0; // range for logtable: log(1+exp(x))  -25 <= x <= 0

const float64_t CMath::NOT_A_NUMBER    	=  NAN;
const float64_t CMath::INFTY            =  INFINITY;	// infinity
const float64_t CMath::ALMOST_INFTY		=  +1e+300;		//a large number
const float64_t CMath::ALMOST_NEG_INFTY =  -1e+300;
const float64_t CMath::PI=M_PI;
const float64_t CMath::MACHINE_EPSILON=DBL_EPSILON;
const float64_t CMath::MAX_REAL_NUMBER=DBL_MAX;
const float64_t CMath::MIN_REAL_NUMBER=DBL_MIN;
const float32_t CMath::F_MAX_VAL32=FLT_MAX;
const float32_t CMath::F_MIN_NORM_VAL32=FLT_MIN;
const float64_t CMath::F_MAX_VAL64=DBL_MAX;
const float64_t CMath::F_MIN_NORM_VAL64=DBL_MIN;
const float32_t CMath::F_MIN_VAL32=(FLT_MIN * FLT_EPSILON);
const float64_t CMath::F_MIN_VAL64=(DBL_MIN * DBL_EPSILON);

#ifdef USE_LOGCACHE
float64_t* CMath::logtable = NULL;
#endif
uint32_t CMath::seed = 0;

CMath::CMath()
: CSGObject()
{
#ifdef USE_LOGCACHE
    LOGRANGE=CMath::determine_logrange();
    LOGACCURACY=CMath::determine_logaccuracy(LOGRANGE);
    CMath::logtable=SG_MALLOC(float64_t, LOGRANGE*LOGACCURACY);
    init_log_table();
#else
	int32_t i=0;
	while ((float64_t)std::log(1 + ((float64_t)exp(-float64_t(i)))))
		i++;

	LOGRANGE=i;
#endif
}

CMath::~CMath()
{
#ifdef USE_LOGCACHE
	SG_FREE(CMath::logtable);
	CMath::logtable=NULL;
#endif
}

#ifdef USE_LOGCACHE
int32_t CMath::determine_logrange()
{
    int32_t i;
    float64_t acc=0;
    for (i=0; i<50; i++)
	{
		acc=((float64_t)log(1+((float64_t)exp(-float64_t(i)))));
		if (acc<=(float64_t)LOG_TABLE_PRECISION)
			break;
	}

    SG_SINFO("determined range for x in table log(1+exp(-x)) is:%d (error:%G)\n",i,acc)
    return i;
}

int32_t CMath::determine_logaccuracy(int32_t range)
{
    range=MAX_LOG_TABLE_SIZE/range/((int)sizeof(float64_t));
    SG_SINFO("determined accuracy for x in table log(1+exp(-x)) is:%d (error:%G)\n",range,1.0/(double) range)
    return range;
}

//init log table of form log(1+exp(x))
void CMath::init_log_table()
{
  for (int32_t i=0; i< LOGACCURACY*LOGRANGE; i++)
	  logtable[i] = std::log(1 + exp(float64_t(-i) / float64_t(LOGACCURACY)));
}
#endif

void CMath::sort(int32_t *a, int32_t cols, int32_t sort_col)
{
  int32_t changed=1;
  if (a[0]==-1) return;
  while (changed)
  {
      changed=0; int32_t i=0;
      while ((a[(i+1)*cols]!=-1) && (a[(i+1)*cols+1]!=-1)) // to be sure
	  {
		  if (a[i*cols+sort_col]>a[(i+1)*cols+sort_col])
		  {
			  for (int32_t j=0; j<cols; j++)
				  CMath::swap(a[i*cols+j],a[(i+1)*cols+j]);
			  changed=1;
		  };
		  i++;
	  };
  };
}

void CMath::sort(float64_t *a, int32_t* idx, int32_t N)
{
	int32_t changed=1;
	while (changed)
	{
		changed=0;
		for (int32_t i=0; i<N-1; i++)
		{
			if (a[i]>a[i+1])
			{
				swap(a[i],a[i+1]) ;
				swap(idx[i],idx[i+1]) ;
				changed=1 ;
			} ;
		} ;
	} ;

}

float64_t CMath::Align(
	char* seq1, char* seq2, int32_t l1, int32_t l2, float64_t gapCost)
{
  float64_t actCost=0 ;
  int32_t i1, i2 ;
  float64_t* const gapCosts1 = SG_MALLOC(float64_t,  l1 );
  float64_t* const gapCosts2 = SG_MALLOC(float64_t,  l2 );
  float64_t* costs2_0 = SG_MALLOC(float64_t,  l2 + 1 );
  float64_t* costs2_1 = SG_MALLOC(float64_t,  l2 + 1 );

  // initialize borders
  for( i1 = 0; i1 < l1; ++i1 ) {
    gapCosts1[ i1 ] = gapCost * i1;
  }
  costs2_1[ 0 ] = 0;
  for( i2 = 0; i2 < l2; ++i2 ) {
    gapCosts2[ i2 ] = gapCost * i2;
    costs2_1[ i2+1 ] = costs2_1[ i2 ] + gapCosts2[ i2 ];
  }
  // compute alignment
  for( i1 = 0; i1 < l1; ++i1 ) {
    swap( costs2_0, costs2_1 );
    actCost = costs2_0[ 0 ] + gapCosts1[ i1 ];
    costs2_1[ 0 ] = actCost;
    for( i2 = 0; i2 < l2; ++i2 ) {
      const float64_t actMatch = costs2_0[ i2 ] + ( seq1[i1] == seq2[i2] );
      const float64_t actGap1 = costs2_0[ i2+1 ] + gapCosts1[ i1 ];
      const float64_t actGap2 = actCost + gapCosts2[ i2 ];
      const float64_t actGap = min( actGap1, actGap2 );
      actCost = min( actMatch, actGap );
      costs2_1[ i2+1 ] = actCost;
    }
  }

  SG_FREE(gapCosts1);
  SG_FREE(gapCosts2);
  SG_FREE(costs2_0);
  SG_FREE(costs2_1);

  // return the final cost
  return actCost;
}

void CMath::linspace(float64_t* output, float64_t start, float64_t end, int32_t n)
{
	float64_t delta = (end-start) / (n-1);
	float64_t v = start;
	index_t i = 0;
	while ( v <= end )
	{
		output[i++] = v;
		v += delta;
	}
	output[n-1] = end;
}

int CMath::is_nan(double f)
{
  return std::isnan(f);
}

int CMath::is_infinity(double f)
{
  return std::isinf(f);
}

int CMath::is_finite(double f)
{
  return std::isfinite(f);
}

bool CMath::strtof(const char* str, float32_t* float_result)
{
	ASSERT(str);
	ASSERT(float_result);

	SGVector<char> buf(strlen(str)+1);

	for (index_t i=0; i<buf.vlen-1; i++)
		buf[i]=tolower(str[i]);
	buf[buf.vlen-1]='\0';

	if (strstr(buf, "inf") != NULL)
	{
		*float_result = CMath::INFTY;

		if (strchr(buf,'-') != NULL)
			*float_result *= -1;
		return true;
	}

	if (strstr(buf, "nan") != NULL)
	{
		*float_result = CMath::NOT_A_NUMBER;
		return true;
	}

	char* endptr = buf.vector;
	*float_result=::strtof(str, &endptr);
	return endptr != buf.vector;
}

bool CMath::strtod(const char* str, float64_t* double_result)
{
	ASSERT(str);
	ASSERT(double_result);

	SGVector<char> buf(strlen(str)+1);

	for (index_t i=0; i<buf.vlen-1; i++)
		buf[i]=tolower(str[i]);
	buf[buf.vlen-1]='\0';

	if (strstr(buf, "inf") != NULL)
	{
		*double_result = CMath::INFTY;

		if (strchr(buf,'-') != NULL)
			*double_result *= -1;
		return true;
	}

	if (strstr(buf, "nan") != NULL)
	{
		*double_result = CMath::NOT_A_NUMBER;
		return true;
	}

	char* endptr = buf.vector;
	*double_result=::strtod(str, &endptr);
	return endptr != buf.vector;
}

bool CMath::strtold(const char* str, floatmax_t* long_double_result)
{
	ASSERT(str);
	ASSERT(long_double_result);

	SGVector<char> buf(strlen(str)+1);

	for (index_t i=0; i<buf.vlen-1; i++)
		buf[i]=tolower(str[i]);
	buf[buf.vlen-1]='\0';

	if (strstr(buf, "inf") != NULL)
	{
		*long_double_result = CMath::INFTY;

		if (strchr(buf,'-') != NULL)
			*long_double_result *= -1;
		return true;
	}

	if (strstr(buf, "nan") != NULL)
	{
		*long_double_result = CMath::NOT_A_NUMBER;
		return true;
	}

	char* endptr = buf.vector;

// fall back to double on win32 / cygwin since strtold is undefined there
#if defined(WIN32) || defined(__CYGWIN__)
	*long_double_result=::strtod(str, &endptr);
#else
	*long_double_result=::strtold(str, &endptr);
#endif

	return endptr != buf.vector;
}

float64_t CMath::get_abs_tolerance(float64_t true_value, float64_t rel_tolerance)
{
	REQUIRE(rel_tolerance > 0 && rel_tolerance < 1.0,
		"Relative tolerance (%f) should be less than 1.0 and positive\n", rel_tolerance);
	REQUIRE(is_finite(true_value),
		"The true_value should be finite\n");
	float64_t abs_tolerance = rel_tolerance;
	if (abs(true_value)>0.0)
	{
		if (std::log(abs(true_value)) + std::log(rel_tolerance) <
		    std::log(F_MIN_VAL64))
			abs_tolerance = F_MIN_VAL64;
		else
			abs_tolerance = abs(true_value * rel_tolerance);
	}
	return abs_tolerance;
}
