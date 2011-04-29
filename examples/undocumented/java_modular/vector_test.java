public class vector_test {
	static {
		System.loadLibrary("Features");
	}

	public static void main(String argv[]) {
		FeaturesJNI.init_shogun__SWIG_4();
		double[] vec = {1,2,3,4}; 
		Labels x = new Labels(vec);
		double[] y = x.get_labels();
		for (int i = 0; i < 4; i++) {
			System.out.println(y[i]);
		}
		FeaturesJNI.exit_shogun();
	}
}
