import os
import logging
import tensorflow as tf
logger = logging.getLogger(__name__)

checkpoint_dir = 'checkpoints'
abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
print(f"Absolute checkpoint path: {abs_checkpoint_dir}")

if not os.path.exists(abs_checkpoint_dir):
    print(f"Checkpoint directory '{abs_checkpoint_dir}' does not exist.")
else:
    print(f"Checkpoint directory '{abs_checkpoint_dir}' exists.")

# List files in the directory
print(f"Files in checkpoint directory: {os.listdir(abs_checkpoint_dir)}")

latest_checkpoint = tf.train.latest_checkpoint(abs_checkpoint_dir)
print(f"Latest checkpoint found: {latest_checkpoint}")
