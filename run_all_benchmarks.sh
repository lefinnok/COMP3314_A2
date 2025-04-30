#!/bin/bash

# MLP Benchmarking Runner Script
# This script runs all the defined benchmark batches

# Exit on error
set -e

# Set the data directory
DATA_DIR="./data"
BENCHMARK_SCRIPT="./benchmarking/benchmark.py"
RESULTS_DIR="./benchmarking/results"

echo "Starting MLP benchmarking runs..."

# Create results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Run all batch tests
echo "Running learning rate benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_learning_rate.yaml --data-dir $DATA_DIR

echo "Running activation function benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_activation_function.yaml --data-dir $DATA_DIR

echo "Running data augmentation benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_data_augmentation.yaml --data-dir $DATA_DIR

echo "Running input mode benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_input_mode.yaml --data-dir $DATA_DIR

echo "Running network architecture benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_network_architechture.yaml --data-dir $DATA_DIR

echo "Running optimizer benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_optimizer.yaml --data-dir $DATA_DIR

echo "Running regularization benchmarks..."
python $BENCHMARK_SCRIPT --batch benchmarking/experiments/batch_regularization.yaml --data-dir $DATA_DIR

echo "All benchmarks completed. Results are available in $RESULTS_DIR"