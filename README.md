# ILR-Net: Low-Light Image Enhancement Network based on the Combination of Iterative Learning Mechanism and Retinex Theory 
Images captured in nighttime or low-light environments are often affected by external factors such as noise and lighting. Aiming at the existing image enhancement algorithms tend to overly 
focus on increasing brightness, while neglecting the enhancement of color and detailed features. This paper proposes a low-light image enhancement network based on a combination of iterative 
learning mechanisms and Retinex theory (defined as ILR-Net) to enhance both detail and color features simultaneously. Specifically, the network continuously learns local and global features of 
low-light images across different dimensions and receptive fields to achieve a clear and convergent illumination estimation. Meanwhile, the denoising process is applied to the reflection component 
after Retinex decomposition to enhance the image's rich color features. Finally, the enhanced image is obtained by concatenating the features along the channel dimension. In the adaptive learning sub
network, a dilated convolution module, U-Net feature extraction module, and adaptive iterative learning module are designed. These modules respectively expand the network's receptive field to 
capture multi-dimensional features, extract the overall and edge details of the image, and adaptively enhance features at different stages of convergence. The Retinex decomposition sub-network 
focuses on denoising the reflection component before and after decomposition to obtain a low-noise, clear reflection component. Additionally, an efficient feature extraction module—global feature 
attention is designed to address the problem of feature loss. Experiments were conducted on six common datasets and in real-world environments. The proposed method achieved PSNR and SSIM 
values of 23.7624dB and 0.8653 on the LOL dataset, and 26.8252dB and 0.7784 on the LOLv2Real dataset, demonstrating significant advantages over other algorithms. 
