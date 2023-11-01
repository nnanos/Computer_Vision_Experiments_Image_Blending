from Pyramids_implementation import *
import cv2




dims = input("Give the dimensions of the gaussian kernel FOR THE IMAGES\n ( expected structure rows,cols ):")
dims = tuple(map(int, dims.split(','))) 
var = input("Give the variance of the gaussian kernel FOR THE IMAGES:")

dims1 = input("Give the dimensions of the gaussian kernel FOR THE MASK\n ( expected structure rows,cols ):")
dims1 = tuple(map(int, dims1.split(','))) 
var1 = input("Give the variance of the gaussian kernel FOR THE MASK:")






def create_mask_for_the_objects(cords):
    zero_arr = np.zeros((2448,3264))
    zero_arr[cords[0]:cords[1],cords[2]:cords[3]]=1
    
    return zero_arr

def replace_zeros_with_mean(img_cropped  , original_img):
    tmp = (img_cropped==0)*1.0
    tmp = tmp*np.mean(original_img.flatten())
    
    return img_cropped + tmp 

#gaussian window
h = matlab_style_gauss2D(dims,int(var))
h_mask = matlab_style_gauss2D(dims1,int(var1))

#mean window
#h = np.ones((20,20))*1/(15*15)

dog1 = cv2.imread('dog1.jpg',0).astype(np.float64)
dog2 = cv2.imread('dog2.jpg',0).astype(np.float64)
cat = cv2.imread('cat.jpg',0).astype(np.float64)
P200 = cv2.imread('P200.jpg',0).astype(np.float64)
bench = cv2.imread('bench.jpg',0).astype(np.float64)
quasimoto = cv2.imread('avatars-000571442352-nxul4r-t500x500.jpeg',0).astype(np.float64)
rows,cols = dog1.shape



dog1_cords = (650,1700,1200,2450)
dog2_cords = (650,1840,1200,2350)
cat_cords = (600,1800,1000,2600)
bench_cords = (900,2000,600,3100)
quasi_cords = (65,145,85,185)

dog1_mask = create_mask_for_the_objects(dog1_cords)
dog2_mask = create_mask_for_the_objects(dog2_cords)
cat_mask = create_mask_for_the_objects(cat_cords)
bench_mask = create_mask_for_the_objects(bench_cords)
quasi_mask = create_mask_for_the_objects(quasi_cords)


dog1_cropped = dog1_mask*dog1

A = np.float32([[1/3,0,100],[0,1/3,500]])
dog1_new = cv2.warpAffine(dog1_cropped,A,(cols,rows))
A = np.float32([[1,0,-300],[0,1,1050]])
dog1_new = cv2.warpAffine(dog1_new,A,(cols,rows))
dog1_mask = dog1_new>0
dog1_mask = (dog1_mask*1).astype(np.float64)





#finding the position where i want to place the bench and then applying an apropriate geometric transformation(translation and scaling)
A = np.float32([[2/3,0,300],[0,2/3,1000]])
bench_new = cv2.warpAffine(bench,A,(cols,rows))
bench_mask = bench_new>0
bench_mask = (bench_mask*1).astype(np.float64)



dog2_cropped = dog2_mask*dog2

A = np.float32([[1/3,0,100],[0,1/3,500]])
dog2_new = cv2.warpAffine(dog2_cropped,A,(cols,rows))
A = np.float32([[1,0,1000],[0,1,1300]])
dog2_new = cv2.warpAffine(dog2_new,A,(cols,rows))
dog2_mask = dog2_new>0
dog2_mask = (dog2_mask*1).astype(np.float64)



cat_cropped = cat_mask*cat
M = cv2.getRotationMatrix2D(((cols//2)-500,(rows//2)+370),180,1/3)
cat_new = cv2.warpAffine(cat_cropped,M,(cols,rows))
cat_mask = cat_new>0
cat_mask = (cat_mask*1).astype(np.float64)

#A = np.float32([[1/3,0,100],[0,1/3,500]])
#cat_new = cv2.warpAffine(cat_cropped,A,(cols,rows))


tmp = np.zeros((rows,cols))
tmp[:quasimoto.shape[0],:quasimoto.shape[1]] = quasimoto
quasimoto_new = tmp
A = np.float32([[1,0,1500],[0,1,1400]])
quasimoto_new = cv2.warpAffine(quasimoto_new,A,(cols,rows))
quasimoto_new[quasimoto_new==255.]=0.
quasi_mask = quasimoto_new>0
quasi_mask = (quasi_mask*1).astype(np.float64)



'''
A = np.float32([[2/3,0,100],[0,2/3,0]])
quasimoto_new = cv2.warpAffine(quasimoto,A,(quasimoto.shape[1],quasimoto.shape[0]))
'''





#REPLACING THE BLACK PIXELS AROUND THE MAKSED IMAGE_NEW WITH THE MEAN OF THE BLENDED WITHOUT THE TECHNIQUE IMAGE TO GET
#A SMOTHER TRANSITION (EDGE) WHEN GETTING THE LAPLACIAN OF THAT IMAGE...
a = bench_new + (1 - bench_mask)*P200
bench_new = replace_zeros_with_mean(bench_new,a)

b = dog1_new + (1-dog1_mask)*a
dog1_new = replace_zeros_with_mean(dog1_new,b)

c = dog2_new + (1-dog2_mask)*b
dog2_new = replace_zeros_with_mean(dog2_new,c)

d = cat_new + (1-cat_mask)*c
cat_new = replace_zeros_with_mean(cat_new,d)

e = quasimoto_new + (1-quasi_mask)*d
quasimoto_new = replace_zeros_with_mean(quasimoto_new,e)



#BLENDING-------------------------------------------------------------------------------------------------------------

pyramid_lvl = 7

#blend P200 with bench----------------------------------------------------------------------------------------



Gm1_bench = get_arbitary_level_of_gaus_pyr(bench_mask,h_mask,pyramid_lvl)

#obtaining the laplacian pyramids from the gaussian
G_bench = get_arbitary_level_of_gaus_pyr(bench_new,h,pyramid_lvl)
G_P200 = get_arbitary_level_of_gaus_pyr(P200,h,pyramid_lvl)  
L_bench = get_lapl_pyr(G_bench,h)
L_P200 = get_lapl_pyr(G_P200,h)      

gaus_dicts = (G_bench,G_P200)
lapl_dicts = (L_bench,L_P200)
B_dict_bech = get_B_dict(Gm1_bench,gaus_dicts,lapl_dicts,pyramid_lvl)
recon1 = reconstruct_blended_img(B_dict_bech,h)
#----------------------------------------------------------------------------------------------------------------

#blend result1 with dog1------------------------------------------------------------------------------------------


Gm1_dog1 = get_arbitary_level_of_gaus_pyr(dog1_mask,h_mask,pyramid_lvl)

#obtaining the laplacian pyramids from the gaussian
G_res1 = get_arbitary_level_of_gaus_pyr(recon1[pyramid_lvl-1],h,pyramid_lvl)
G_dog1 = get_arbitary_level_of_gaus_pyr(dog1_new,h,pyramid_lvl)  
L_res1 = get_lapl_pyr(G_res1,h)
L_dog1 = get_lapl_pyr(G_dog1,h)      

gaus_dicts = (G_dog1,G_res1)
lapl_dicts = (L_dog1,L_res1)
B_dict_dog1 = get_B_dict(Gm1_dog1,gaus_dicts,lapl_dicts,pyramid_lvl)
recon2 = reconstruct_blended_img(B_dict_dog1,h)
#-----------------------------------------------------------------------------------------------------------------


#blend result2 with dog2------------------------------------------------------------------------------------------


Gm1_dog2 = get_arbitary_level_of_gaus_pyr(dog2_mask,h_mask,pyramid_lvl)

#obtaining the laplacian pyramids from the gaussian
G_res2 = get_arbitary_level_of_gaus_pyr(recon2[pyramid_lvl-1],h,pyramid_lvl)
G_dog2 = get_arbitary_level_of_gaus_pyr(dog2_new,h,pyramid_lvl)  
L_res2 = get_lapl_pyr(G_res2,h)
L_dog2 = get_lapl_pyr(G_dog2,h)      

gaus_dicts = (G_dog2,G_res2)
lapl_dicts = (L_dog2,L_res2)
B_dict_dog2 = get_B_dict(Gm1_dog2,gaus_dicts,lapl_dicts,pyramid_lvl)
recon3 = reconstruct_blended_img(B_dict_dog2,h)
#-----------------------------------------------------------------------------------------------------------------


#blend result3 with cat------------------------------------------------------------------------------------------


Gm1_cat = get_arbitary_level_of_gaus_pyr(cat_mask,h_mask,pyramid_lvl)

#obtaining the laplacian pyramids from the gaussian
G_res3 = get_arbitary_level_of_gaus_pyr(recon3[pyramid_lvl-1],h,pyramid_lvl)
G_cat = get_arbitary_level_of_gaus_pyr(cat_new,h,pyramid_lvl)  
L_res3 = get_lapl_pyr(G_res3,h)
L_cat = get_lapl_pyr(G_cat,h)      

gaus_dicts = (G_cat,G_res3)
lapl_dicts = (L_cat,L_res3)
B_dict_cat = get_B_dict(Gm1_cat,gaus_dicts,lapl_dicts,pyramid_lvl)
recon4 = reconstruct_blended_img(B_dict_cat,h)
#-----------------------------------------------------------------------------------------------------------------


#blend result4 with my image--------------------------------------------------------------------------------------
Gm1_quas = get_arbitary_level_of_gaus_pyr(quasi_mask,h_mask,pyramid_lvl)

#obtaining the laplacian pyramids from the gaussian
G_res4 = get_arbitary_level_of_gaus_pyr(recon4[pyramid_lvl-1],h,pyramid_lvl)
G_quas = get_arbitary_level_of_gaus_pyr(quasimoto_new,h,pyramid_lvl)  
L_res4 = get_lapl_pyr(G_res4,h)
L_quas = get_lapl_pyr(G_quas,h)      

gaus_dicts = (G_quas,G_res4)
lapl_dicts = (L_quas,L_res4)
B_dict_quas = get_B_dict(Gm1_quas,gaus_dicts,lapl_dicts,pyramid_lvl)
recon5 = reconstruct_blended_img(B_dict_quas,h)

plt.imshow(recon5[pyramid_lvl-1],cmap="gray")
plt.show()    
