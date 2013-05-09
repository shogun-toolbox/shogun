#ifndef DSFMT_PARAMS44497_H
#define DSFMT_PARAMS44497_H

/* #define DSFMT_N	427 */
/* #define DSFMT_MAXDEGREE	44536 */
#define DSFMT_POS1	304
#define DSFMT_SL1	19
#define DSFMT_MSK1	UINT64_C(0x000ff6dfffffffef)
#define DSFMT_MSK2	UINT64_C(0x0007ffdddeefff6f)
#define DSFMT_MSK32_1	0x000ff6dfU
#define DSFMT_MSK32_2	0xffffffefU
#define DSFMT_MSK32_3	0x0007ffddU
#define DSFMT_MSK32_4	0xdeefff6fU
#define DSFMT_FIX1	UINT64_C(0x75d910f235f6e10e)
#define DSFMT_FIX2	UINT64_C(0x7b32158aedc8e969)
#define DSFMT_PCV1	UINT64_C(0x4c3356b2a0000000)
#define DSFMT_PCV2	UINT64_C(0x0000000000000001)
#define DSFMT_IDSTR	"dSFMT2-44497:304-19:ff6dfffffffef-7ffdddeefff6f"


/* PARAMETERS FOR ALTIVEC */
#if defined(__APPLE__)	/* For OSX */
    #define ALTI_SL1 	(vector unsigned char)(3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
    #define ALTI_SL1_PERM \
	(vector unsigned char)(2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1)
    #define ALTI_SL1_MSK \
	(vector unsigned int)(0xffffffffU,0xfff80000U,0xffffffffU,0xfff80000U)
    #define ALTI_MSK	(vector unsigned int)(DSFMT_MSK32_1, \
			DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4)
#else	/* For OTHER OSs(Linux?) */
    #define ALTI_SL1 	{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}
    #define ALTI_SL1_PERM \
	{2,3,4,5,6,7,30,30,10,11,12,13,14,15,0,1}
    #define ALTI_SL1_MSK \
	{0xffffffffU,0xfff80000U,0xffffffffU,0xfff80000U}
    #define ALTI_MSK \
	{DSFMT_MSK32_1, DSFMT_MSK32_2, DSFMT_MSK32_3, DSFMT_MSK32_4}
#endif

#endif /* DSFMT_PARAMS44497_H */
