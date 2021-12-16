
import numpy as np
import matplotlib.pyplot as plt
from numba import jit,prange
from skimage import filters
from scipy.ndimage import gaussian_filter


def noise_f(ty,inten,image):
    '''
    Creates noisy images with different types of noise and different intensities.

    Parameters
    ----------
    ty : Type of noise function, can be 'white', 'gaussian', or 
    'impulsive' (also accepts 'sp' and 'salt and pepper').
    inten : Intensity of the noise.
    image : Image to be added noise.

    Returns
    -------
    noisy_image : Yielded noisy image.
    ty.capitalize(): Capitalized word representing the type of noise added.

    '''
    
    ty = ty.lower() # Making the code insensitive to caps
    
    rows,cols = image.shape[0],image.shape[1]
    
    if ty == 'gaussian':
        
        noise = np.random.normal(loc=0, scale=inten/100, size=image.shape)
        noisy_image = np.clip((image + noise),0,1)
        
    elif ty == 'white':
        
        noise = np.random.uniform(low=0, high=1, size=image.shape)
        noisy_image = np.clip((image + noise*inten/100),0,1)
        
    elif ty in ('impulsive','salt and pepper','sp'):
        
        noisy_image = np.copy(image) # Duplicating the image so that it is not overwritten
        ty = 'impulsive'
        
        for i in range(inten*100):    
            
            b_w = np.random.choice([0,1]) # Randomly selects 0 or 1
            
            y = np.random.randint(0, rows-1) # Randomly selected pixels
            x = np.random.randint(0, cols-1)
            
            if b_w == 0: # If b_w is 0, a random pixel is set to 0
                noisy_image[y][x] = 0
            elif b_w == 1: # If b_w is 1, a random pixel is set to 1
                noisy_image[y][x] = 1
    else:
        print('That is not a valid type of noise.') # Making sure wrong noises are called
        return None
                
    return noisy_image, ty.capitalize()


@jit(nopython=True, parallel=True)
def NLM_method(img, pad_img, mode='normal', h = 1, D = 1, alpha = 1):
    '''
    Performs the Non-local Means method on an input 
    image so that it yields a filtered image.    

    Parameters
    ----------
    img : Input image.
    pad_img : Padded input image.
    mode : Optional; the default is 'normal'. This parameter 
    forces the function to filter the image differently, based 
    on either the 'normal' NLM or on different modifications:
        - 'cpp': CPP method, which takes into consideration
         the similarity with central pixels.
         - 'maximum': This modification assigns the maximum
         similarity value to the comparison between a pixel and
         itself.
    h : Optional; the default is 1. Parameter to optimize when 
    using either one of the methods.
    D : Optional; the default is 1. Parameter to optimize when 
    using the CPP modification.
    alpha : Optional; the default is 1. Parameter to optimize 
    when using the CPP modification.

    Returns
    -------
    NLM_img : NLM-filtered image.

    '''
    
    NLM_img=np.zeros(img.shape)
    for i in prange(1,pad_img.shape[0]-1): 
        
        for j in prange(1,pad_img.shape[1]-1):  
            
            patch1=pad_img[i-1:i+2,j-1:j+2] # Patch centered in the pixel to be filtered
            raw_ws = []  
            
            for k in prange(1,pad_img.shape[0]-1): 
                
                for l in prange(1,pad_img.shape[1]-1):
                    
                    patch2 = pad_img[k-1:k+2,l-1:l+2] # patch 1 and patch 2 are np arrays

                    euclid = np.sqrt(np.sum((patch1-patch2)**2)) # Unique for each i,j
                    raw_weight = np.exp(-euclid/(h**2))
                    
                    if mode =='cpp' and (i != k and j != l):
                        # If the mode is 'cpp', the weights take into consideration 
                        # the similarity with central pixels
                        n = 1 / (1+((np.abs(img[i-1][j-1]-img[k-1][l-1]) / D) ** (2*alpha)))
                        raw_weight = n * raw_weight 
                        
                    raw_ws.append(raw_weight)
            
            raw_ws = np.reshape(np.array(raw_ws),img.shape)
            
            if mode == 'maximum':
                # If the mode is 'maximum', the weights of a pixel compared to 
                # itself is set to the maximum similarity value for that pixel
                raw_ws[i-1][j-1] = np.max(raw_ws)

            # All patch2 have been added; proceed to NL for pixel i (pad_img[i][j])
            
            norm_ws = raw_ws/np.sum(raw_ws)  # Normalizing the raw weights
            # Overwritting the NLM image with the new NLM-filtered pixel values:
            NLM_img[i-1][j-1] = np.sum(np.multiply(norm_ws,img))  
    
    return NLM_img


def anisodiff(original_img, threshold, sigma, niter):
    '''
    Performs greater smoothing (filtering) within a region than with 
    neighboring regions (edges), reducing the noise but keeping 
    the details.

    Parameters
    ----------
    original_img : Input image, that will be the noisy image that the 
    noise_f function returns.
    threshold: Certain gradient value chosen as threshold that will 
    decide whether to apply or not filtering.
    sigma : Standard deviation of the gaussian filter that will decide 
    how the smoothing will be.
    niter : Number of times this procedure is applied to the image.

    Returns
    -------
    sobel_img : Gradient image.
    gauss_img: Gaussian filtered image.
    img_out: Image after applying anisotropic diffusion filtering.

    '''
    for ii in range(niter):
        image = np.copy(original_img)
        sobel_img = filters.sobel(image)
        gauss_img = gaussian_filter(image, sigma)
        img_out=np.zeros(image.shape)
        for i in range(0,image.shape[0]):
            for j in range(1,image.shape[1]):
                grad = sobel_img[i,j] 
                if grad >= threshold:
                    img_out[i,j] = image[i,j]
                else:
                    img_out[i,j] = gauss_img[i,j]
        original_img=np.copy(img_out)
    return sobel_img,gauss_img, img_out


def plot_img(images,titles,cmap=None):
    '''
    Plots multiple images.

    Parameters
    ----------
    images : Images to plot.
    titles : Title of the plots of each image.
    cmap : Color map of the plots.

    Returns
    -------
    None. The function is called and x plots are plotted.

    '''
    
    plt.figure(figsize=(12, 12))
    ind=1
    
    for i in images:
        plt.subplot(1,len(titles),ind)
        plt.imshow(i, cmap=cmap)
        plt.title(titles[ind-1]), plt.axis('off')
        ind+=1
    
    plt.tight_layout()
    plt.show()
     
    return None






