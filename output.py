# import os
# import numpy as np
# import torch
# import stylegan2
# from stylegan2 import utils
# import cv2 as cv
# from google.colab.patches import cv2_imshow
# from tqdm import tqdm
# dir = 'latent'

# G_blend = 'G_blend.pth'
# # G_blend = 'Gs.pth'

# def synthesis(G_file, latent_file):
#     device = torch.device(0)
#     G = stylegan2.models.load(G_file).G_synthesis 
#     latent = np.load(latent_file, allow_pickle=True)
#     G.to(device)
#     latent = torch.tensor(latent[np.newaxis, ...]).to(device)
#     out = G(latent)
#     out = utils.tensor_to_PIL(out, pixel_min=-1, pixel_max=1)[0]
#     return out

# for l in tqdm(sorted(os.listdir(dir))):
#   if l[-3:] != 'npy':
#     continue
#   latent = os.path.join(dir, l)
#   out = synthesis(G_blend, latent)
#   print(l)
#   display(out)

# out.save('output.png')


import os
import numpy as np
import torch
import stylegan2
from stylegan2 import utils
import cv2 as cv
from tqdm import tqdm
from PIL import Image  # To handle image display and saving

dir = 'latent'
G_blend = 'G_blend.pth'
# G_blend = 'Gs.pth'

def synthesis(G_file, latent_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = stylegan2.models.load(G_file).G_synthesis 
    latent = np.load(latent_file, allow_pickle=True)
    G.to(device)
    latent = torch.tensor(latent[np.newaxis, ...]).to(device)
    out = G(latent)
    out = utils.tensor_to_PIL(out, pixel_min=-1, pixel_max=1)[0]
    return out

# Ensure output directory exists
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Iterate through latent files and process each
for l in tqdm(sorted(os.listdir(dir))):
    if l[-3:] != 'npy':
        continue
    latent = os.path.join(dir, l)
    out = synthesis(G_blend, latent)
    print(f"Processing file: {l}")
    
    # Show image using OpenCV
    out_cv = np.array(out)[:, :, ::-1]  # Convert PIL image to OpenCV format (BGR)
    cv.imshow("Generated Image", out_cv)
    cv.waitKey(0)  # Display until a key is pressed
    cv.destroyAllWindows()
    
    # Save image
    output_path = os.path.join(output_dir, f"{os.path.splitext(l)[0]}.png")
    out.save(output_path)
    print(f"Saved generated image to {output_path}")

