#ifndef DSFMT_PARAMS1279_H
#define DSFMT_PARAMS1279_H

#include <shogun/lib/config.h>

/* #define DSFMT_N	12 */
/* #define DSFMT_MAXDEGREE	1376 */
#define DSFMT_POS1	9
#define DSFMT_SL1	19
#define DSFMT_MSK1	UINT64_C(0x000efff7ffddffee)
#define DSFMT_MSK2	UINT64_C(0x000fbffffff77fff)
#define DSFMT_MSK32_1	0x000efff7U
#define DSFMT_MSK32_2	0xffddffeeU
#define DSFMT_MSK32_3	0x000fbfffU
#define DSFMT_MSK32_4	0xfff77fffU
#define DSFMT_FIX1	UINT64_C(0xb66627623d1a31be)
#define DSFMT_FIX2	UINT64_C(0x04b6c51147b6109b)
#define DSFMT_PCV1	UINT64_C(0x7049f2da382a6aeb)
#define DSFMT_PCV2	UINT64_C(0xde4ca84a40000001)
#define DSFMT_IDSTR	"dSFMT2-1279:9-19:efff7ffddffee-fbffffff77fff"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1	(vector unsigned char)(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffffU,0xfff80000U,0xffffffffU,0xfff80000U)
    #define ALTI_MSK	(vector unsigned int)(DSFMT_MSK32_1, \
			DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1	{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}
    #define ALTI_SL1_PERM \
	{2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1}
    #define ALTI_SL1_MSK \
	{0xffffffffU,0xfff80000U,0xffffffffU,0xfff80000U}
    #define ALTI_MSK \
	{DSFMT_MSK32_1, DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4}
#endif

#endif /* DSFMT_PARAMS1279_H */
