rm -rf results/network/*

export CS231N_PLOT_BATCH=True
export CS231N_DATASET_LIMIT=100
export CS231N_MINING_RATIO=1
export CS231N_BATCH_SZ=32
export CS231N_NUM_EPOCHS=1000
export CS231N_LEARNING_RATE=5e-3
export CS231N_PCT_VALIDATION=10.0
export CS231N_POOLING_TYPE=l2
export CS231N_INIT_STDDEV=0.05

python training_v3.py
