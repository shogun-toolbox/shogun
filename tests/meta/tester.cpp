#include <shogun/base/init.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/lib/WrappedObjectArray.h>

#include <iostream>
#include <sstream>

using namespace shogun;
using namespace std;

int main(int argc, const char *argv[])
{
    if (argc != 6)
        cout << "Usage: tester REL_DIR NAME TARGET GENERATED_RESULTS_DIR REFERENCE_RESULTS_DIR" << endl;
        
    string rel_dir = argv[1];
    string name = argv[2];
    string target = argv[3];
    string generated_results_dir = argv[4];
    string reference_results_dir = argv[5];
    
    stringstream os;
    os << generated_results_dir << "/" << target << "/" << rel_dir << "/" << name;
    string fname_full_generated = os.str();
    os.str("");
    os << reference_results_dir << "/" << rel_dir << "/" << name;
    string fname_full_reference = os.str();

    init_shogun_with_defaults();
    CSerializableAsciiFile* f = new CSerializableAsciiFile(fname_full_generated.c_str(), 'r');
    CSerializableAsciiFile* f_ref = new CSerializableAsciiFile(fname_full_reference.c_str(), 'r');
    SG_REF(f);
    SG_REF(f_ref);
    
    CWrappedObjectArray* a = new CWrappedObjectArray();
    CWrappedObjectArray* a_ref = new CWrappedObjectArray();
    SG_REF(a);
    SG_REF(a_ref);
    
    a->load_serializable(f);
    a_ref->load_serializable(f_ref);
    
    bool equal = a->equals(a_ref);
    
    SG_UNREF(f);
    SG_UNREF(f_ref);
    SG_UNREF(a);
    SG_UNREF(a_ref);
    
    exit_shogun();
    return equal != true;
}

