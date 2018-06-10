rm -rf results/network/*

export CS231N_PLOT_BATCH=False
export CS231N_DATASET_LIMIT=400
export CS231N_BATCH_SZ=128
export CS231N_NUM_EPOCHS=1000
export CS231N_OVERRIDE_LEARNING_RATE=5e-4
export CS231N_PCT_VALIDATION=10.0
export CS231N_INIT_STDDEV=1
#export CS231N_NUM_FILTERS_CONV1=
#export CS231N_NUM_FILTERS_CONV2=
#export CS231N_CONV_CONNECTIVITY=
#export CS231N_SAVED_MODEL_PREFIX=sess
#export CS231N_SAVED_STATS_DIR=results/train_stats_06-01/
export CS231N_DEVICE_NAME=cpu

#mkdir $CS231N_SAVED_STATS_DIR
python binary.py