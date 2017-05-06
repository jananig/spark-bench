# for gen_data.sh;  200M data size = 1 million points
NUM_OF_EXAMPLES=500 #00000
NUM_OF_FEATURES=6
#NUM_OF_PARTITIONS=120 #0

# for run.sh
NUM_OF_CLASS_C=2
NUM_OF_TREES_C=3
featureSubsetStrategy="auto"
impurityC="gini"
maxDepthC=5
maxBinsC=32
seedC=12345
modeC="Classification"

NUM_OF_CLASS_R=10
NUM_OF_TREES_R=3
featureSubsetStrategy="auto"
impurityR="variance"
maxDepthR=5
maxBinsR=100
seedR=12345
modeR="Regression"

MAX_ITERATION=3

SPARK_STORAGE_MEMORYFRACTION=0.79

