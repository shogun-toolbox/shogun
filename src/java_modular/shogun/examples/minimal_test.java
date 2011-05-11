package shogun.examples;
import shogun.*;

public class minimal_test {
    static {
        System.loadLibrary("Kernel");
    }

    public static void main(String argv[]) {
        KernelJNI.init_shogun__SWIG_4();
        GaussianKernel x = new GaussianKernel();
        System.out.println(x.get_width());
        KernelJNI.exit_shogun();
    }
}


