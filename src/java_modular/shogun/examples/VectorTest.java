public class VectorTest {
	static {
		System.loadLibrary("Features");
	}

	public static void main(String argv[]) {
		Features.init_shogun();
		double[] vec = {1,2,3,4}; 
		Labels x = new Labels(vec);
		double[] y = new double[4];
		x.get_labels(y);
		for (int i = 0; i < 4; i++) {
			System.out.println(y[i]);
		}
		Features.exit_shogun();
	}
}
