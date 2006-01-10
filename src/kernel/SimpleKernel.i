%module SimpleKernel
%{
   #include "kernel/SimpleKernel.h" 
%}

%include "Kernel.i"
%include "swigfiles/common.i"

%feature("notabstract") SimpleKernel;

template <class ST> class CSimpleKernel : public CKernel
{
	public:
		CSimpleKernel(LONG cachesize) : CKernel(cachesize)
		{
		}

		virtual bool init(CFeatures* l, CFeatures* r, bool do_init)
		{
			CKernel::init(l,r,do_init);

			if ( (l->get_feature_class() != C_SIMPLE) ||
					(r->get_feature_class() != C_SIMPLE) ||
					((CSimpleFeatures<ST>*) l)->get_num_features() != ((CSimpleFeatures<ST>*) r)->get_num_features() )
			{
				CIO::message(M_ERROR, "train or test features not of type SIMPLE, or #features mismatch (l:%d vs. r:%d)\n",
						((CSimpleFeatures<ST>*) l)->get_num_features(),((CSimpleFeatures<ST>*) l)->get_num_features());
			}
			return true;
		}

		inline virtual EFeatureClass get_feature_class() { return C_SIMPLE; }
};

%template(CSimpleCharKernel) CSimpleKernel<CHAR>;
%template(CSimpleIntKernel) CSimpleKernel<INT>;
