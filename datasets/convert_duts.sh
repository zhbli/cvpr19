#!/bin/bash
# Usage:
#   bash ./convert_duts.sh

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./DUTS"

# Root path for DUTS dataset.
DUTS_ROOT=${WORK_DIR}

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

TRAIN_IMAGE_FOLDER="${DUTS_ROOT}/DUTS-TR/DUTS-TR-Image"
TRAIN_SEMANTIC_SEG_FOLDER="${DUTS_ROOT}/DUTS-TR/DUTS-TR-Mask"
TRAIN_SUPERPIXEL_FOLDER="${DUTS_ROOT}/DUTS-TR/DUTS-TR-Superpixel"
TEST_IMAGE_FOLDER="${DUTS_ROOT}/DUTS-TE/DUTS-TE-Image"
TEST_SEMANTIC_SEG_FOLDER="${DUTS_ROOT}?DUTS-TE/DUTS-TE-Mask"
TEST_SUPERPIXEL_FOLDER="${DUTS_ROOT}/DUTS-TE/DUTS-TE-Superpixel"
TRAIN_LIST_FOLDER="${DUTS_ROOT}/train_list"
TEST_LIST_FOLDER="${DUTS_ROOT}/test_lsit"

echo "Converting DUTS train dataset..."
python ./build_duts_data.py \
    --image_folder="${TRAIN_IMAGE_FOLDER}" \
    --semantic_segmentation_folder="${TRAIN_SEMANTIC_SEG_FOLDER}" \
    --superpixel_folder="${TRAIN_SUPERPIXEL_FOLDER}" \
    --list_folder="${TRAIN_LIST_FOLDER}" \
    --image_format="jpg" \
    --output_dir="${OUTPUT_DIR}"

echo "Converting DUTS test dataset..."
python ./build_duts_data.py \
    --image_folder="${TEST_IMAGE_FOLDER}" \
    --semantic_segmentation_folder="${TEST_SEMANTIC_SEG_FOLDER}" \
    --superpixel_folder="${TEST_SUPERPIXEL_FOLDER}" \
    --list_folder="${TEST_LIST_FOLDER}" \
    --image_format="jpg" \
    --output_dir="${OUTPUT_DIR}"