package RandomForests.src.main.java;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import java.util.HashMap;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;

public class RandomForestsApp {

    public static void main(String[] args) {

            Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);
            Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
            
            if (args.length < 4) {
                System.out.println("usage: <input> <output> <numClass> <numTrees>"
                        + " <featureSubsetStrategy> <impurity> <maxDepth> <maxBins> <seed> <mode: Regression/Classification>");
                    System.exit(0);
            }

            String input = args[0];
            String output = args[1];
            Integer numClasses = Integer.parseInt(args[2]);
            Integer numTrees = Integer.parseInt(args[3]);

            String featureSubsetStrategy = args[4];
            String impurity = args[5];//"gini";
            Integer maxDepth = Integer.parseInt(args[6]);
            Integer maxBins = Integer.parseInt(args[7]);
            Integer seed = Integer.parseInt(args[8]);
            String mode=args[9];
                
            SparkConf sparkConf = new SparkConf().setAppName("JavaRandomForestClassificationApp");
            JavaSparkContext jsc = new JavaSparkContext(sparkConf);


            // Cache the data since we will use it again to compute training error.
            long start = System.currentTimeMillis();

            //load and parse the data file            
            JavaRDD<String> tmpdata = jsc.textFile(input);
                
            JavaRDD<LabeledPoint> data = tmpdata.map(
                    new Function<String, LabeledPoint>() {
                        public LabeledPoint call(String line) {                        
                            return LabeledPoint.parse(line);
                        }
                    }
            );

            // Split the data into training and test sets (30% held out for testing)
            JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
            JavaRDD<LabeledPoint> trainingData = splits[0];
            JavaRDD<LabeledPoint> testData = splits[1];
            
            double loadTime = (double)(System.currentTimeMillis() - start) / 1000.0;

            
            HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();

            RandomForestModel tmpModel;
            if(mode.equals("Classification")){
              tmpModel = RandomForest.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);
            } else{
              tmpModel = RandomForest.trainRegressor(trainingData,
                  categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
            }

            double trainingTime = (double)(System.currentTimeMillis() - start) / 1000.0;
          
            start = System.currentTimeMillis();
            final RandomForestModel model = tmpModel;
 
          // Evaluate model on test instances and compute test error
          JavaPairRDD<Double, Double> predictionAndLabel =
            testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
              @Override
              public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<>(model.predict(p.features()), p.label());
              }
            });

          Double testErr =
            1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
              @Override
              public Boolean call(Tuple2<Double, Double> pl) {
                return !pl._1().equals(pl._2());
              }
            }).count() / testData.count();
        
        double testTime = (double)(System.currentTimeMillis() - start) / 1000.0;

         System.out.printf("{\"loadTime\":%.3f,\"trainingTime\":%.3f,\"testTime\":%.3f}\n", 
              loadTime, trainingTime, testTime);

        System.out.println("Test Error: " + testErr);

        System.out.println("Learned " + mode + " forest model:\n" + model.toDebugString());

        // Save and load model
        model.save(jsc.sc(), "target/tmp/myRandomForest" + mode + "Model"); 
        RandomForestModel sameModel = RandomForestModel.load(jsc.sc(),
          "target/tmp/myRandomForest" + mode + "Model");
        jsc.stop();

    }
}