%module CharKernel
%{
   #include "kernel/CharKernel.h" 
%}

%include "kernel/SimpleKernel.i"
typedef char CHAR;

class CCharKernel : public CSimpleKernel<CHAR>
{
	public:
		CCharKernel(LONG cachesize) : CSimpleCharKernel(cachesize) {}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CSimpleCharKernel::init(l,r, do_init);
			assert(l->get_feature_type()==F_CHAR);
			assert(r->get_feature_type()==F_CHAR);
			return true;
		}

		inline virtual EFeatureType get_feature_type() { return F_CHAR; }
};
