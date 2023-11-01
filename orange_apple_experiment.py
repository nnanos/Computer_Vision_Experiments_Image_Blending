from Pyramids_implementation import *
import cv2







dims = input("Give the dimensions of the gaussian kernel FOR THE IMAGES\n ( expected structure rows,cols ):")
dims = tuple(map(int, dims.split(','))) 
var = input("Give the variance of the gaussian kernel FOR THE IMAGES:")

dims1 = input("Give the dimensions of the gaussian kernel FOR THE MASK\n ( expected structure rows,cols ):")
dims1 = tuple(map(int, dims1.split(','))) 
var1 = input("Give the variance of the gaussian kernel FOR THE MASK:")









#gray scale case
orange = cv2.imread('orange.jpg',0)
apple = cv2.imread('apple.jpg',0)



#filter kernel
h = matlab_style_gauss2D(dims,int(var))
h_mask = matlab_style_gauss2D(dims1,int(var1))


m1 = np.ones(apple.shape).astype(np.float64)
m1[:,apple.shape[1]//2:] = 0
m2 = 1 - m1

pyramid_lvl = 4

#obtaining the laplacian pyramids from the gaussian
G_orange = get_arbitary_level_of_gaus_pyr(orange,h,pyramid_lvl)
G_apple = get_arbitary_level_of_gaus_pyr(apple,h,pyramid_lvl)
L_orange = get_lapl_pyr(G_orange,h)
L_apple = get_lapl_pyr(G_apple,h)

Gm1 = get_arbitary_level_of_gaus_pyr(m1,h_mask,pyramid_lvl)

gaus_dicts = (G_apple,G_orange)
lapl_dicts = (L_apple,L_orange)
B_dict = get_B_dict(Gm1,gaus_dicts,lapl_dicts,pyramid_lvl)

recon = reconstruct_blended_img(B_dict,h)

#ploting---------------------------------------------------------------------------------------------------


#gaussian pyr for mask
f, axarr = plt.subplots(3,2,figsize=(10,10))
f.tight_layout()
axarr[0,0].imshow(Gm1["level0"] , cmap="gray" )
axarr[0,0].title.set_text('Gaussian_level_0')
axarr[0,1].imshow( Gm1["level1"] , cmap="gray")
axarr[0,1].title.set_text('Gaussian_level_1')
axarr[1,0].imshow(Gm1["level2"] , cmap="gray")
axarr[1,0].title.set_text('Gaussian_level_2')
axarr[1,1].imshow( Gm1["level3"] , cmap="gray")
axarr[1,1].title.set_text('Gaussian_level_3')
axarr[2,0].imshow( Gm1["level4"] , cmap="gray")
axarr[2,0].title.set_text('Gaussian_level_4')

#laplacian pyr for images
f, axarr = plt.subplots(3,2,figsize=(10,10))
f.tight_layout()
axarr[0,0].imshow(L_apple["level0"] , cmap="gray" )
axarr[0,0].title.set_text('laplacian_level_0')
axarr[0,1].imshow( L_apple["level1"] , cmap="gray")
axarr[0,1].title.set_text('laplacian_level_1')
axarr[1,0].imshow(L_apple["level2"] , cmap="gray")
axarr[1,0].title.set_text('laplacian_level_2')
axarr[1,1].imshow( L_apple["level3"] , cmap="gray")
axarr[1,1].title.set_text('laplacian_level_3')
axarr[2,0].imshow( G_apple["level4"] , cmap="gray")
axarr[2,0].title.set_text('Gaussian_level_4')

f, axarr = plt.subplots(3,2,figsize=(10,10))
f.tight_layout()
axarr[0,0].imshow(L_orange["level0"] , cmap="gray" )
axarr[0,0].title.set_text('laplacian_level_0')
axarr[0,1].imshow( L_orange["level1"] , cmap="gray")
axarr[0,1].title.set_text('laplacian_level_1')
axarr[1,0].imshow(L_orange["level2"] , cmap="gray")
axarr[1,0].title.set_text('laplacian_level_2')
axarr[1,1].imshow( L_orange["level3"] , cmap="gray")
axarr[1,1].title.set_text('laplacian_level_3')
axarr[2,0].imshow( G_orange["level4"] , cmap="gray")
axarr[2,0].title.set_text('Gaussian_level_4') 

f, axarr = plt.subplots(3,2,figsize=(10,10))
f.tight_layout()
axarr[0,0].imshow(B_dict["b0"] , cmap="gray" )
axarr[0,0].title.set_text("blended_laplacians_with\nsmooth_transition_lvl0")
axarr[0,1].imshow( B_dict["b1"] , cmap="gray")
axarr[0,1].title.set_text("blended_laplacians_with\nsmooth_transition_lvl1")
axarr[1,0].imshow(B_dict["b2"] , cmap="gray")
axarr[1,0].title.set_text("blended_laplacians_with\nsmooth_transition_lvl2")
axarr[1,1].imshow( B_dict["b3"] , cmap="gray")
axarr[1,1].title.set_text("blended_laplacians_with\nsmooth_transition_lvl3")
axarr[2,0].imshow( B_dict["g_0_L"] , cmap="gray")
axarr[2,0].title.set_text('blended_gaussians_with\nsmooth_transition_lvl4')
axarr[2,1].imshow( recon[pyramid_lvl-1] , cmap="gray")
axarr[2,1].title.set_text('reconstructed_image')
plt.subplots_adjust(hspace=0.3)


'''
plotting_pyramid(pyramid_lvl,Gm1,"gaussian")
plotting_pyramid(pyramid_lvl,L_apple,"laplacian")
plotting_pyramid(pyramid_lvl,L_orange,"laplacian")
plotting_pyramid(pyramid_lvl,B_dict,"B_dict")
'''

plt.show()