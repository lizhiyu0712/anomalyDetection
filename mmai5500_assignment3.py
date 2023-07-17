# 1. Load the data
# Extract video frames as images
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

def convert_video_to_images(img_folder, filename='assignment2_video.avi'):
  """
  Converts the video file (assignment2_video.avi) to JPEG images.
  Once the video has been converted to images, then this function doesn't
  need to be run again.
  Arguments
  ---------
  filename : (string) file name (absolute or relative path) of video file.
  img_folder : (string) folder where the video frames will be
  stored as JPEG images.
  """
  # Make the img_folder if it doesn't exist.'
  try:
    if not os.path.exists(img_folder):
      os.makedirs(img_folder)
  except OSError:
    print('Error')

  # Make sure that the abscense/prescence of path
  # separator doesn't throw an error.
  img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
  # Instantiate the video object.
  video = cv2.VideoCapture(filename)

  # Check if the video is opened successfully
  if not video.isOpened():
    print("Error opening video file")
  i = 0
  while video.isOpened():
    ret, frame = video.read()
    if ret:
      im_fname = f'{img_folder}frame{i:0>4}.jpg'
      print('Captured...', im_fname)
      cv2.imwrite(im_fname, frame)
      i += 1
    else:
      break

  video.release()
  cv2.destroyAllWindows()
  if i:
    print(f'Video converted\n{i} images written to {img_folder}')

# Load the extracted image files
from glob import glob
import numpy as np

def load_images(img_dir, im_width=60, im_height=44):
  """
  Reads, resizes and normalizes the extracted image frames from a folder.
  The images are returned both as a Numpy array of flattened images
  Arguments
  ---------
  img_dir : (string) the directory where the images are stored.
  im_width : (int) The desired width of the image.
  The default value works well.
  im_height : (int) The desired height of the image.
  The default value works well.
  Returns
  X : (numpy.array) An array of the flattened images.
  images : (list) A list of the resized images.
  """
  images = []
  fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
  fnames.sort()

  for fname in fnames:
    im = Image.open(fname)
    # resize the image to im_width and im_height.
    im_array = np.array(im.resize((im_width, im_height)))
    # Convert uint8 to decimal and normalize to 0 - 1.
    images.append(im_array.astype(np.float32) / 255.)
    # Close the PIL image once converted and stored.
    im.close()

  # Flatten the images to a single vector
  X = np.array(images).reshape(-1, np.prod(images[0].shape))

  return X, images

convert_video_to_images("/content/images", filename='assignment3_video.avi')

load_images("/content/images", im_width=60, im_height=44)

# 2. Builds up and trains the autoencoder.
# autocoder with three fully connected layers
import keras
from keras import layers
from keras.layers import Input, Dense
from keras.models import Model

def build_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    # Decoder
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_shape[0], activation='sigmoid')(decoded)

    # Autoencoder
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

# Normalize the images
image_folder = "/content/images"
X, images = load_images(image_folder)
X = X.astype('float32') / 255.

# Define the input shape based on the image dimensions
input_shape = X.shape[1:]

# Build the autoencoder model
autoencoder = build_autoencoder(input_shape)

# Train the autoencoder
autoencoder.fit(X, X, epochs=10, batch_size=32, validation_split=0.2)

# 3. Measures the reconstruction loss for all frames.
# Reconstruct the images using the trained autoencoder
reconstructed_images = autoencoder.predict(X)

# Calculate the reconstruction loss for all frames
reconstruction_loss = np.mean(np.square(X - reconstructed_images), axis=1)

# Print the reconstruction loss for each frame
for i, loss in enumerate(reconstruction_loss):
    print(f"Frame {i}: Reconstruction Loss = {loss}")

# Calculate the average reconstruction loss across all frames
average_loss = np.mean(reconstruction_loss)
print(f"Average Reconstruction Loss: {average_loss}")

# 4. Set a threshold
threshold = np.mean(reconstruction_loss) + 0.2 * np.std(reconstruction_loss)

# 5. Function to find anomalous
def  predict(frame):
    # Reshape and normalize the frame
    normalized_frame = frame.reshape(1, -1).astype('float32') / 255.
    # Reconstruct the frame using the autoencoder
    reconstructed_frame = autoencoder.predict(normalized_frame)
    # Calculate the reconstruction loss for the frame
    frame_loss = np.mean(np.square(normalized_frame - reconstructed_frame))
    # Check if the frame loss exceeds the threshold
    anomaly = frame_loss > threshold
    return anomaly

# Example usage
frame_index = 729
# Index of the frame to check for anomaly
frame = images[frame_index]  # Get the frame image
is_anomalous = predict(frame)  # Check if the frame is anomalous

print(f"Frame {frame_index}: Anomalous = {is_anomalous}")

