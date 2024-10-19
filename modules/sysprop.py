import tensorflow as tf # type: ignore
import os, warnings

# Ocultar advertencias y logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Solo errores
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Establecer nivel de log para TensorFlow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)