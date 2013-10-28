#ifndef DSFMT_PARAMS521_H
#define DSFMT_PARAMS521_H

/* #define DSFMT_N	4 */
/* #define DSFMT_MAXDEGREE	544 */
#define DSFMT_POS1	3
#define DSFMT_SL1	25
#define DSFMT_MSK1	UINT64_C(0x000fbfefff77efff)
#define DSFMT_MSK2	UINT64_C(0x000ffeebfbdfbfdf)
#define DSFMT_MSK32_1	0x000fbfefU
#define DSFMT_MSK32_2	0xff77efffU
#define DSFMT_MSK32_3	0x000ffeebU
#define DSFMT_MSK32_4	0xfbdfbfdfU
#define DSFMT_FIX1	UINT64_C(0xcfb393d661638469)
#define DSFMT_FIX2	UINT64_C(0xc166867883ae2adb)
#define DSFMT_PCV1	UINT64_C(0xccaa588000000000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-521:3-25:fbfefff77efff-ffeebfbdfbfdf"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1	(vector unsigned char)(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(3,4,5,6,7,29,29,29,11,12,13,14,15,0,1,2)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffffU,0xfe000000U,0xffffffffU,0xfe000000U)
    #define ALTI_MSK	(vector unsigned int)(DSFMT_MSK32_1, \
			DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1	{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}
    #define ALTI_SL1_PERM \
	{3,4,5,6,7,29,29,29,11,12,13,14,15,0,1,2}
    #define ALTI_SL1_MSK \
	{0xffffffffU,0xfe000000U,0xffffffffU,0xfe000000U}
    #define ALTI_MSK \
	{DSFMT_MSK32_1, DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4}
#endif

#endif /* DSFMT_PARAMS521_H */
