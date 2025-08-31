import tensorflow as tf

model = tf.keras.models.load_model('rice_disease_model_2.h5')  # works in TF 2.x

# Save as .keras
model.save('rice_disease_model_2.keras')