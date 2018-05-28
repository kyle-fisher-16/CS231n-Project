rm -rf results/network/*

export CS231N_PLOT_BATCH=True
export CS231N_DATASET_LIMIT=1000
export CS231N_MINING_RATIO=8
export CS231N_BATCH_SZ=128
export CS231N_NUM_EPOCHS=1000
export CS231N_LEARNING_RATE=5e-5
export CS231N_PCT_VALIDATION=15.0

python training_v2.py