# Animefy yourself!

### Pre-requisite

python version 3.10.8
CUDA Toolkit 11.8
torch: 2.2.1+cu118

Download pretrain model on FFHQ `./G_pretrain.pth` ([google drive](https://drive.google.com/uc?id=19vr5taIJSbNMiqihW1-vWSZkXUK7tdPR)) .
Download other pretrain model `./G_out.pth` ([google drive](https://drive.google.com/file/d/1J6sJaRZJg4dAoSw03fyanWV2oEscOeSk/view?usp=sharing))

# Blend Two Models into one i.e G_blend.pth (keep the directory same)

python blend_models.py --model1 path/to/G_pretrain.pth --model2 path/to/G_out.pth --output path/to/G_blend.pth

Take an input image and save it to raw folder (make sure you have created directory raw, aligned/test/class_1 and latent)

# Run the following command to generate aligned face crop image in the following directory

python align_images.py raw aligned/test/class_1/

# Generate results in .npy in the latent dir

python run_projector.py project_real_images --network=G_pretrain.pth --data_dir=aligned/test/ --output=latent/ --num_steps=400 --gpu=0

# process latent dir file and display output images

python output.py
