from PIL import Image
image = Image.open("Test.jpg")
image2 = Image.open("Test2.jpg")

from transformers import pipeline

checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

prediction1 = depth_estimator(image)
prediction2 = depth_estimator(image2)

# prediction1["depth"].show()
# prediction2["depth"].show()

import torch

x = prediction2["predicted_depth"]

total = torch.sum(x).item()
size = x.size(dim=1) * x.size(dim=2)

estimated_distance = total / size

print(estimated_distance)






