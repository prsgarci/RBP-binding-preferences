#get model predictions for test sequences
predictions = model.predict(x_test)

# Get the top num_plots predictions
num_plots = 10

# Get the sorted indices
sorted_indices = np.argsort(predictions[:, 0])[::-1]

# Extract the top num_plots sequences
X = x_test[sorted_indices[:num_plots]]

# Reshape X to (num_plots, 200, 4)
X = X.reshape((num_plots, 200, 4))




#########################################################
import tensorflow as tf

@tf.function
def calculate_saliency_map(X, model, class_index=0):
  """fast function to generate saliency maps"""
  if not tf.is_tensor(X):
    X = tf.Variable(X)

  with tf.GradientTape() as tape:
    tape.watch(X)
    output = model(X)[:,class_index]
  return tape.gradient(output, X)

saliency_map = calculate_saliency_map(X, model)
saliency_map = saliency_map.numpy()

#########################################################
import pandas as pd
import logomaker

def plot_saliency_map(scores, alphabet, ax=None):
  L,A = scores.shape
  counts_df = pd.DataFrame(data=0.0, columns=list(alphabet), index=list(range(L)))
  for a in range(A):
    for l in range(L):
      counts_df.iloc[l,a] = scores[l,a]

  if not ax:
    ax = plt.subplot(1,1,1)
  logomaker.Logo(counts_df, ax=ax)


saliency_scores = saliency_map * X
for scores in saliency_scores:
  fig = plt.figure(figsize=(20,1))
  ax = plt.subplot(1,1,1)
  plot_saliency_map(scores, alphabet, ax)