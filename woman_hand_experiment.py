from Pyramids_implementation import *
import cv2




dims = input("Give the dimensions of the gaussian kernel FOR THE IMAGES\n ( expected structure rows,cols ):")
dims = tuple(map(int, dims.split(','))) 
var = input("Give the variance of the gaussian kernel FOR THE IMAGES:")

dims1 = input("Give the dimensions of the gaussian kernel FOR THE MASK\n ( expected structure rows,cols ):")
dims1 = tuple(map(int, dims1.split(','))) 
var1 = input("Give the variance of the gaussian kernel FOR THE MASK:")




#filter kernel
h = matlab_style_gauss2D(dims,int(var))
h_mask = matlab_style_gauss2D(dims1,int(var1))

woman = cv2.imread('woman.png',0)
hand = cv2.imread('hand.png',0)



#making the hands dimmension equal to womans
hand = np.insert(hand, 0 , 0 ,axis=0)
hand = np.delete(hand , slice(195,200) , axis=1)

eye_cropped = woman[83:115,70:130]

mask_eye = np.zeros(woman.shape)
mask_eye[83:115,70:130]=1

plt.imshow( mask_eye*woman + (1-mask_eye)*hand, cmap="gray")


pyramid_lvl = 2

Gm1 = get_arbitary_level_of_gaus_pyr(mask_eye,h_mask,pyramid_lvl)

#obtaining the laplacian pyramids from the gaussian
G_woman = get_arbitary_level_of_gaus_pyr(woman,h,pyramid_lvl)
G_hand = get_arbitary_level_of_gaus_pyr(hand,h,pyramid_lvl)
L_woman = get_lapl_pyr(G_woman,h)
L_hand = get_lapl_pyr(G_hand,h)    

gaus_dicts = (G_woman,G_hand)
lapl_dicts = (L_woman,L_hand)
B_dict = get_B_dict(Gm1,gaus_dicts,lapl_dicts,pyramid_lvl)

recon = reconstruct_blended_img(B_dict,h)

plt.imshow(recon[pyramid_lvl-1],cmap="gray")
plt.show()