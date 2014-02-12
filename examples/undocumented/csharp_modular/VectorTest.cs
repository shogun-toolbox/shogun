using System;

public class VectorTest
{
	public static void Main(string[] args) {
		modshogun.init_shogun_with_defaults();

		double[] y = new double[5] {0, 1, 2, 3, 4,};
		MulticlassLabels x = new MulticlassLabels(y);
		double[] r = x.get_labels();
		for (int i = 0; i < 5; i++) {
					Console.WriteLine(r[i]);
		}

	}
}
