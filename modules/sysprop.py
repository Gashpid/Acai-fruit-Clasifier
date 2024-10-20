import tensorflow as tf # type: ignore
import os, warnings

# Hide TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Solo errores
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set log level for TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)