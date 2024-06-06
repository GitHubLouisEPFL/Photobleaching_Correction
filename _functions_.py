import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm
from scipy.optimize import curve_fit
import pickle
from scipy.signal import convolve2d
import matplotlib.animation as animation
from IPython.display import HTML
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-----------------------------------------------Training functions-------------------------------------------------#

def train_N2K(model, noisy_tensor, loss_function, optimizer, 
              scheduler, number_of_iter = 50000, track_progress = True, 
              sampling__freq_val_losses = 1000):
    '''
    Train the model with the Noise2Kernel method

    Input:
        model : the model to train (nn.Module)
        noisy_tensor : the noisy grayscale image as a tensor (torch.Tensor)
        loss_function : the loss function (nn.Module)
        optimizer : the optimizer (torch.optim)
        scheduler : the scheduler (torch.optim.lr_scheduler)
        number_of_iter : number of iterations (int)
        track_progress : boolean, if True, the progress is tracked (bool)
        sampling__freq_val_losses : the frequency at which the validation loss is computed (int)

    Output:
        model : the trained model (nn.Module)
        losses : the training losses (list of floats)
        val_losses : the validation losses (list of floats)
        all_images : the denoised images (list of numpy arrays)
    '''

    model = model.to(device)
    noisy_tensor = noisy_tensor.to(device)

    if track_progress:
        iterator = tqdm(range(number_of_iter))
    else:
        iterator = range(number_of_iter)

    losses = []
    val_losses = []
    all_images = []

    for i in iterator:

        #################################################### TRAINING ####################################################

        model.train() # Set the model to training mode
        
        net_output = model(noisy_tensor) # Get the output of the network
        loss = loss_function(net_output, noisy_tensor) # Compute the loss
        losses.append(loss.item())

        optimizer.zero_grad() # Reset the gradient 
        loss.backward() # Compute the gradient
        optimizer.step() # Update the weights

        if (i+1 % 1000 == 0):
            scheduler.step()

        #################################################### VALIDATION ####################################################

        with torch.no_grad():
            if (i % sampling__freq_val_losses == 0):

                net_output = torch.zeros_like(noisy_tensor) # only zeros, same size as image

                for j in range(100):
                    net_output += model(noisy_tensor)
                    
                net_output = net_output/100

                val_loss = loss_function(net_output, noisy_tensor)
                val_losses.append(val_loss.item())

                all_images.append(net_output)

    return model, losses, val_losses, net_output


def train_N2K_Custom(model, noisy_tensor, loss_function, optimizer, 
              scheduler, number_of_iter = 50000, track_progress = True, 
              sampling__freq_val_losses = 1000, epoch=0):
    '''
    Train the model with the Noise2Kernel method

    Input:
        model : the model to train (nn.Module)
        noisy_tensor : the noisy grayscale image as a tensor (torch.Tensor)
        loss_function : the loss function (nn.Module)
        optimizer : the optimizer (torch.optim)
        scheduler : the scheduler (torch.optim.lr_scheduler)
        number_of_iter : number of iterations (int)
        track_progress : boolean, if True, the progress is tracked (bool)
        sampling__freq_val_losses : the frequency at which the validation loss is computed (int)

    Output:
        model : the trained model (nn.Module)
        losses : the training losses (list of floats)
        val_losses : the validation losses (list of floats)
        all_images : the denoised images (list of numpy arrays)
    '''

    model = model.to(device)
    noisy_tensor = noisy_tensor.to(device)

    if track_progress:
        iterator = tqdm(range(number_of_iter))
    else:
        iterator = range(number_of_iter)

    losses = []
    val_losses = []
    all_images = []

    for i in iterator:

        #################################################### TRAINING ####################################################

        model.train() # Set the model to training mode
        
        net_output = model(noisy_tensor) # Get the output of the network
        loss = loss_function(net_output, noisy_tensor, epoch) # Compute the loss
        losses.append(loss.item())

        optimizer.zero_grad() # Reset the gradient 
        loss.backward() # Compute the gradient
        optimizer.step() # Update the weights

        if (i+1 % 1000 == 0):
            scheduler.step()

        #################################################### VALIDATION ####################################################

        with torch.no_grad():
            if (i % sampling__freq_val_losses == 0):

                net_output = torch.zeros_like(noisy_tensor) # only zeros, same size as image

                for j in range(100):
                    net_output += model(noisy_tensor)
                    
                net_output = net_output/100

                val_loss = loss_function(net_output, noisy_tensor, epoch)
                val_losses.append(val_loss.item())

                all_images.append(net_output)

    return model, losses, val_losses, net_output



def train_N2S(model, noisy_tensor, masker, loss_function, optimizer, number_of_iter = 500, track_progress = False):
    '''
    Train the model with the Noise2Self method

    Input:
        model : the model to train (nn.Module)
        noisy_tensor : the noisy grayscale image as a tensor (torch.Tensor)
        masker : the masker (Masker object)
        loss_function : the loss function (nn.Module)
        optimizer : the optimizer (torch.optim)
        number_of_iter : number of iterations (int)
        track_progress : boolean, if True, the progress is tracked (bool)

    Output:
        model : the trained model (nn.Module)
        losses : the training losses (list of floats)
        val_losses : the validation losses (list of floats)
        all_images : the denoised images (list of numpy arrays)

    '''

    model = model.to(device)
    noisy_tensor = noisy_tensor.to(device)

    if track_progress:
        iterator = range(number_of_iter)
    else:
        iterator = tqdm(range(number_of_iter))

    losses = []
    val_losses = []
    all_images = []

    for i in iterator:

        #################################################### TRAINING ####################################################

        model.train() # Set the model to training mode
        
        net_input, mask = masker.mask(noisy_tensor, i % (masker.n_masks - 1)) # Get the input of the network and the mask
        net_output = model(net_input) # Get the output of the network
        
        loss = loss_function(net_output*mask, noisy_tensor*mask) # Compute the loss
        optimizer.zero_grad() # Reset the gradient 
        loss.backward() # Compute the gradient
        optimizer.step() # Update the weights


        #################################################### VALIDATION ####################################################

        losses.append(loss.item())
        model.eval()

        net_input, mask = masker.mask(noisy_tensor, masker.n_masks - 1)
        net_output = model(net_input)     

        denoised = np.clip(model(noisy_tensor).detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64) # Clip the values to [0, 1] and convert to numpy

        ### Save results and print if needed ###

        val_loss = loss_function(net_output*mask, noisy_tensor*mask)
        val_losses.append(val_loss.item())

        all_images.append(denoised)

    return model, losses, val_losses, all_images




def train_N2K_3D(model, tensor_series, loss_function, optimizer, scheduler, number_of_iter = 500, 
                 track_progress = True, sampling__freq_val_losses = 100):
    '''
    Train the model with the Noise2Kernel method for a 3D tensor (temporal dimension included)

    Input:
        model : the model to train (nn.Module)
        tensor_series : the 3D tensor, representing a signal through time (torch.Tensor)
        loss_function : the loss function (nn.Module)
        optimizer : the optimizer (torch.optim)
        scheduler : the scheduler (torch.optim.lr_scheduler)
        number_of_iter : number of iterations (int)
        track_progress : boolean, if True, the progress is tracked (bool)
        sampling__freq_val_losses : the frequency at which the validation loss is computed (int)

    Output:
        model : the trained model (nn.Module)
        losses : the training losses (list of floats)
        val_losses : the validation losses (list of floats)
        all_series : the output of the network (list of tensors)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    tensor_series = tensor_series.to(device)

    if track_progress:
        iterator = tqdm(range(number_of_iter))
    else:
        iterator = range(number_of_iter)

    losses = []
    val_losses = []
    all_series = []

    for i in iterator:

        #################################################### TRAINING ####################################################

        model.train() # Set the model to training mode
        
        net_output = model(tensor_series) # Get the output of the network
        loss = loss_function(net_output, tensor_series) # Compute the loss for the central image (assumuing tensor series has 3 images)
        losses.append(loss.item())

        optimizer.zero_grad() # Reset the gradient 
        loss.backward() # Compute the gradient
        optimizer.step() # Update the weights

        if (i+1 % 1000 == 0):
            scheduler.step()

        #################################################### VALIDATION ####################################################

        with torch.no_grad():
            if (i % sampling__freq_val_losses == 0):

                net_output = torch.zeros_like(tensor_series) # only zeros, same size as tensor_series

                for j in range(100):
                    net_output += model(tensor_series)
                    
                net_output = net_output/100

                val_loss = loss_function(net_output, tensor_series) # Compute the validation loss for the central image (assumuing tensor series has 3 images)
                val_losses.append(val_loss.item())

                all_series.append(net_output)

    return model, losses, val_losses, all_series




#-----------------------------------------------Performances functions-------------------------------------------------#

def PSNR_plot(all_images, clean_grayscale, Save = False, path = None):
    '''
    Plot the PSNR of the denoised images

    Input:
        all_images : denoised images (list of numpy arrays)
        clean_grayscale : reference clean grayscale image (numpy array)
        Save : boolean, if True, the plot is saved (bool)
        path : path where the plot is saved (str)
        
    '''
    plt.plot([compare_psnr(im, clean_grayscale) for im in all_images])
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title('PSNR')
    plt.show()
    if Save:
        plt.savefig(path)

def Losses_plot(losses, val_losses, Save = False, path = None):
    '''
    Plot the training losses and validation losses

    Input:
        all_images : denoised images (list of numpy arrays)
        clean_grayscale : reference clean grayscale image (numpy array)
        Save : boolean, if True, the plot is saved (bool)
        path : path where the plot is saved (str)
    
    '''
    plt.plot(np.log(losses), label='log(losses)', marker='o')
    plt.plot(np.log(val_losses), label='log(val_losses)', marker='s')
    plt.xlabel('Index')
    plt.ylabel('log(y)')
    plt.title('Training losses and Validation Losses')
    plt.legend()
    plt.show()
    if Save:
        plt.savefig(path)


def compare_PSNRs(img1, img2, img_ref, data_range=4095, colorbar=False, return_psnrs=False):
    '''
    Compare the PSNR of two images with respect to a reference image

    Input:
        img1 : the first image (numpy array)
        img2 : the second image (numpy array)
        img_ref : the reference image (numpy array)
        data_range : the maximum value of the pixel values (int)
        colorbar : boolean, if True, a colorbar is displayed (bool)
        return_psnrs : boolean, if True, the PSNRs are returned (bool)

    Output:
        psnr1 : the PSNR of the first image (float)
        psnr2 : the PSNR of the second image (float)
    '''
    psnr1 = compare_psnr(img1, img_ref, data_range=data_range)
    psnr2 = compare_psnr(img2, img_ref, data_range=data_range)
    name_img1 = get_var_name(img1)
    name_img2 = get_var_name(img2)
    # Plot figures side by side with their PSNR
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img1, cmap='gray', vmin=0, vmax=data_range)
    ax[0].set_title(f'PSNR: {psnr1:.2f}')
    if colorbar:
        plt.colorbar(ax[0].imshow(img1, cmap='gray', vmin=0, vmax=data_range), ax=ax[0])
    ax[1].imshow(img2, cmap='gray', vmin=0, vmax=data_range)
    ax[1].set_title(f'PSNR: {psnr2:.2f}')
    if colorbar:
        plt.colorbar(ax[1].imshow(img2, cmap='gray', vmin=0, vmax=data_range), ax=ax[1])
    ax[2].imshow(img_ref, cmap='gray')
    ax[2].set_title('Reference')
    if colorbar:
        plt.colorbar(ax[2].imshow(img_ref, cmap='gray'), ax=ax[2])
    for a in ax:
        a.axis('off')
    plt.show()

    if return_psnrs:
        return psnr1, psnr2 
    

def get_model_size(model_path):

    file_size = os.path.getsize(model_path)
    print('model weights file size in MB:', file_size / 1024 /1024)
    return file_size / 1024 /1024


def get_model_inference_time_3D(model):
    model.to(torch.device('cpu'))
    input = torch.randn(1, 1, 50, 256, 256)

    start = time.time()
    out = model(input)
    end = time.time()
    assert out.shape == torch.Size([1, 1, 50, 256, 256])
    print('model inference time in seconds:', end - start)
    return end - start

def get_model_inference_time_2D(model):
    model.to(torch.device('cpu'))
    input = torch.randn(1, 1, 256, 256)

    start = time.time()
    out = model(input)
    end = time.time()
    assert out.shape == torch.Size([1, 1, 256, 256])
    print('model inference time in seconds:', end - start)
    return end - start


#-----------------------------------------------Saving functions-------------------------------------------------#

def save_variable(variable_to_save, path, file_name, OS='Linux'):
    '''
    Save a variable to a file .pkl

    Input:
        variable_to_save : the variable to save
        path : the path where the file is saved (str)
        file_name : the name of the file (str)
        OS : the operating system (str)
    '''


    if OS == 'Linux':
        file_path = path + '/' + file_name + '.pkl'
    elif OS == 'Windows':
        file_path = path + '\\' + file_name + '.pkl'
    else:
        print('OS not recognized')
        return

    # Open the file in binary mode
    with open(file_path, 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(variable_to_save, file)


def load_variable(file_path):
    '''
    Load a variable from a file .pkl

    Input:
        file_path : the path of the file (str)

    Output:
        variable : the variable loaded
    '''

    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        # Load the variable
        variable = pickle.load(file)

    return variable


#-----------------------------------------------Add Noise functions-------------------------------------------------#

def add_gaussian_noise(target_image, noise_level):
    '''
    Add Gaussian noise to an image

    Input:
        target_image : the image to which noise is added, values between 0 and 1 (numpy array)
        noise_level : the standard deviation of the Gaussian noise (float)

    Output:
        noisy_image : the noisy image (numpy array)
    '''
    noisy_image = target_image + np.random.randn(target_image.shape[0], target_image.shape[1]) * noise_level
    noisy_image[noisy_image < 0] = 0

    return noisy_image

def add_poisson_noise(target_image, scale_factor=1):
    '''
    Add Poisson noise to an image

    Input:
        target_image : the grayscale image to which noise is added, values between 0 and 1 (numpy array)
        scale_factor : the scale factor applied to the image (float)

    Output:
        noisy_image : the noisy image (numpy array)
    '''

    scaled_image = target_image * scale_factor

    noisy_image = np.random.poisson(scaled_image)
    noisy_image[noisy_image < 0] = 0

    return noisy_image




#-----------------------------------------------Simulation functions-------------------------------------------------#

def exponential_decay(t, tau, A=1):
    result = np.where(tau <= 100, A * np.exp(-t/tau), 1)
    return result

def naive_fixed_simulator(image, tau, nb_images, poisson_scale_factor=1, gaussian_noise_level=0.05*4095):
    '''
    Simulate microscopy images for an object that is not moving. The same time constant is used for all pixels.

    Input:
        image : the clean grayscale image (numpy array), encoded in 16 bits, but values between 0 and 4095
        tau : the time constant (float)
        nb_images : the number of images (int)

    Outputs:
        ground_truth : the simulated decaying signal without noise (3D numpy arrays)
        array_sumulated_images : a 3D numpy array containing the simulated noisy images
    '''
    
    ground_truth = np.zeros((nb_images, image.shape[0], image.shape[1]))
    array_simulated_images = np.zeros((nb_images, image.shape[0], image.shape[1]))
    reference_image = image

    for i in range(nb_images):
        # apply exponential decay to pixel values higher than 100
        decaying_signal = exponential_decay(i, tau) * (reference_image)
        poisson_ = add_poisson_noise(decaying_signal, poisson_scale_factor)
        poisson_and_gauss = add_gaussian_noise(poisson_, gaussian_noise_level)

        ground_truth[i] = decaying_signal
        array_simulated_images[i] = poisson_and_gauss

    return ground_truth, array_simulated_images


def personalized_fixed_simulator(image, tau_map, nb_images, poisson_scale_factor=1, gaussian_noise_level=0.05*4095):
    '''
    Simulate microscopy images for an object that is not moving. Each pixel has its own time constant.

    Input:
        image : the clean grayscale image (numpy array), encoded in 16 bits, but values between 0 and 4095
        tau_map : the time constant map (numpy array), can be a scalar, in which case the same time constant is used for all pixels
        nb_images : the number of images (int)
        poissson_scale_factor : the scale factor for the poisson noise (float)
        gaussian_noise_level : the standard deviation of the gaussian noise (float)

    Outputs:
        ground_truth : the simulated decaying signal without noise (3D numpy arrays)
        array_simulated_images : a 3D numpy array containing the simulated noisy images
    '''

    # if tau_map and amplitude_map are scalars, convert to numpy arrays
    if isinstance(tau_map, (int, float)):
        tau_map = np.where(image <= 100, 10000, tau_map)

    # check if the image and the maps have the same size
    assert image.shape == tau_map.shape, "The image and the tau map must have the same size"

    ground_truth = np.zeros((nb_images, image.shape[0], image.shape[1]))
    array_simulated_images = np.zeros((nb_images, image.shape[0], image.shape[1]))
    reference_image = image

    for i in range(nb_images):
        decaying_signal = np.multiply(exponential_decay(i, tau_map), reference_image)
        poisson_ = add_poisson_noise(decaying_signal, poisson_scale_factor)
        poisson_and_gauss = add_gaussian_noise(poisson_, gaussian_noise_level)

        ground_truth[i] = decaying_signal
        array_simulated_images[i] = poisson_and_gauss

    return ground_truth, array_simulated_images


def photobleaching_simulator(image, tau_map, nb_images, speed_level = 5, sigma = 5, poisson_scale_factor=1, gaussian_noise_level=0.05*4095):
    '''
    Simulate microscopy images for an object that is moving, following rigid movements. Time constant can be either a scalar, in which case
    it will be the same for all pixels, or a 2D array, with a specific time constant for each pixel.

    Input:
        image : the clean grayscale image (numpy array), encoded in 16 bits, but values between 0 and 4095
        tau_map : the time constant map (numpy array), can be a scalar, in which case the same time constant is used for all pixels
        nb_images : the number of images (int)
        speed_level : the speed of the simulation, the higher the faster (int between 0 and 100)
        sigma : the standard deviation of the movement direction (degrees)
        poissson_scale_factor : the scale factor for the poisson noise (float)
        gaussian_noise_level : the standard deviation of the gaussian noise (float)

    Outputs:
        ground_truth : the simulated decaying signal without noise (3D numpy arrays)
        array_simulated_images : a 3D numpy array containing the simulated noisy images
    '''

    # if tau_map and amplitude_map are scalars, convert to numpy arrays
    if isinstance(tau_map, (int, float)):
        tau_map = np.ones(image.shape) * tau_map

    # check if the image and the maps have the same size
    assert image.shape == tau_map.shape, "The image and the tau map must have the same size"

    ground_truth = np.zeros((nb_images, image.shape[0], image.shape[1]))
    array_simulated_images = np.zeros((nb_images, image.shape[0], image.shape[1]))
    reference_frame = image

    if speed_level == 0 or int(round((speed_level/100)*image.shape[0]))<2:
        return personalized_fixed_simulator(image, tau_map, nb_images, poisson_scale_factor=poisson_scale_factor, gaussian_noise_level=gaussian_noise_level)
    
    previous_angle = np.random.randint(360)

    poisson_ = add_poisson_noise(reference_frame, poisson_scale_factor)
    poisson_and_gauss = add_gaussian_noise(poisson_, gaussian_noise_level)
    ground_truth[0] = reference_frame
    array_simulated_images[0] = poisson_and_gauss

    for i in range(1, nb_images):

        decaying_signal = np.multiply(exponential_decay(i, tau_map), reference_frame)
        decaying_signal[decaying_signal < 100] = 100
        poisson_ = add_poisson_noise(decaying_signal, poisson_scale_factor)
        poisson_and_gauss = add_gaussian_noise(poisson_, gaussian_noise_level)

        angle = rad2deg(generate_new_angle(deg2rad(previous_angle), deg2rad(sigma)))
        previous_angle = angle

        current_clean_frame = move(decaying_signal, angle, speed_level)
        current_noisy_frame = move(poisson_and_gauss, angle, speed_level)
        reference_frame = move(reference_frame, angle, speed_level)
        tau_map = move(tau_map, angle, speed_level)

        ground_truth[i] = current_clean_frame
        array_simulated_images[i] = current_noisy_frame

    return ground_truth, array_simulated_images




def curve2fit(x,a,b):
    return a * np.exp((-1/b)*x)

def exponential_fit(simulation_array, p0=[4095, 10], real_data=False, _mean_background=300, tau_max=100, t0=0, time_step=1):
    '''
    Fit an exponential decay to each pixel of a of a sequence of images, and return the time constant and the amplitude for each pixel.
    The background's time constant is set to 100 and the amplitude to 100 as well.

    Input:
        simulation_array : the sequence of images (3D numpy array)
        t0 : the initial time (int)
        time_step : the time step between two images (int)

    Outputs:
        tau_map : the time constant map (2D numpy array)
        signal : the amplitude map (2D numpy array)
    '''
    if type(simulation_array) == list:
        simulation_array = list_to_array(simulation_array)

    t = np.arange(t0, simulation_array.shape[0], time_step)
    y = np.zeros(len(t))
    tau_map = np.zeros((simulation_array.shape[1], simulation_array.shape[2]))
    signal = np.zeros((simulation_array.shape[1], simulation_array.shape[2]))

    for i in range(simulation_array.shape[1]):
        for j in range(simulation_array.shape[2]):

            for time in range(len(t)):
                y[time] = simulation_array[time][i][j]

            estimated_amplitude = np.max(y)

            if y[0]>0.2:
                denom = np.log(y[-1]/y[0])
            else:
                denom=0

            if denom!=0:
                estimated_tau = -len(y)/denom
            else:
                estimated_tau = 0.1

            p0 = [estimated_amplitude, estimated_tau]

            if(np.mean(y) > _mean_background):

                if estimated_tau>0.1:
                    popt, _ = curve_fit(curve2fit, t, y, p0=p0, maxfev=10000)
                else:
                    popt = (np.max(y), 0.1) 
                a, tau_fit = popt
                amplitude = a

                tau_map[i, j] = tau_fit
                signal[i, j] = amplitude

            else:

                if real_data:
                    tau_map[i, j] = tau_max
                    signal[i, j] = _mean_background/2

                else:
                    tau_map[i, j] = tau_max
                    signal[i, j] = 100

    if real_data:
        tau_map[tau_map>tau_max]=tau_max
        tau_map[tau_map<0]= tau_max
        tau_map[np.isnan(tau_map)] = tau_max
        signal[signal>np.max(simulation_array)] = np.max(simulation_array)
        signal[signal<0] = 0
    
    else:
        tau_map[tau_map>100]=tau_max
        tau_map[tau_map<0]= tau_max
        tau_map[np.isnan(tau_map)] = tau_max
        signal[signal>4095] = 4095
        signal[signal<0] = 0

    return tau_map, signal



def generate_new_angle(angle, sigma):
    '''
    Generate a new angle by adding a random offset to the initial angle

    Input:
        angle : the initial angle (rad)
        sigma : the standard deviation of the Gaussian noise (rad)

    Output:
        new_angle : the new angle (rad)
    '''
    
    angle_offset = np.random.normal(loc=0, scale=sigma) # add a random offset to the initial angle
    new_angle = angle + angle_offset # new angle
    new_angle = new_angle % (2 * np.pi) # make sure the angle is between 0 and 2*pi
    
    return new_angle

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def choose_direction(deg_angle, kernel_size):
    '''
    Choose the direction of the movement of the object. Transforms an angle information into a convolution kernel. Angle is in degrees.
    A zero angle corresponds to a movement to the top left corner, then it follows counterclockwise.

    Input:
        deg_angle : the angle in degrees (int)
        kernel_size : the size of the convolution kernel (int)

    Output:
        kernel : the convolution kernel (numpy array)
    '''
    
    deg_angle = deg_angle % 360

    kernel = np.zeros((kernel_size, kernel_size))

    max_idx = kernel_size - 1

    angle_step = max_idx/90
    nb_steps = deg_angle*angle_step

    # 4 situations

    if nb_steps < max_idx:
        kernel[int(round(nb_steps)), 0] = 1

    elif max_idx <= nb_steps < 2*max_idx:
        kernel[max_idx, int(round(nb_steps - max_idx))] = 1

    elif 2*max_idx <= nb_steps < 3*max_idx:
        kernel[int(round(3*max_idx - nb_steps)), max_idx] = 1

    elif 3*max_idx <= nb_steps < 4*max_idx:
        kernel[0, int(round(4*max_idx - nb_steps))] = 1
    else:
        raise ValueError("The angle is not in the right range")

    return kernel

def move(image, angle, speed_level):
    '''
    Move an object in an image in a certain direction

    Input:
        image : the image (numpy array)
        angle : the angle of the movement (degrees)
        speed_level : the speed level of the movement (int)

    Output:
        moved_image : the image with the object moved (numpy array)
    '''

    # check that speed level is between 0 and 100
    if speed_level < 0 or speed_level > 100:
        raise ValueError("The speed level must be between 0 and 100")

    image_size = image.shape[0]
    normalized_speed = speed_level/100
    kernel_size = int(round(normalized_speed*image_size))
    kernel = choose_direction(angle, kernel_size)
 
    return convolve2d(image, kernel, mode='same', fillvalue=100)


def moving_object(image, nb_images, speed_level, sigma=30):
    '''
    Simulate a moving object in an image. The object moves in a random direction at each time step, with the specified
    speed level and standard deviation for the direction changes.

    Input:
        image : the image (numpy array)
        nb_images : the number of images (int)
        speed_level : the speed level of the movement (integer between 0 and 100)
        sigma : the standard deviation for direction change (degrees)

    Output:
        list_moving_images : the list of images with the moving object (list of numpy arrays)
    '''

    list_moving_images = []

    if speed_level == 0 or int(round((speed_level/100)*image.shape[0]))<2:
        for i in range(nb_images):
            list_moving_images.append(image.copy())
        return list_moving_images

    previous_angle = np.random.randint(360)
    current_frame = image.copy()
    list_moving_images.append(current_frame.copy())

    for i in range(nb_images - 1):

        # depending on the previous direction, the object will be more likely to continue in the same direction or close to it
        angle = rad2deg(generate_new_angle(deg2rad(previous_angle), deg2rad(sigma)))
        previous_angle = angle
        current_frame = move(current_frame, angle, speed_level)
        list_moving_images.append(current_frame.copy())
        
    return list_moving_images


def define_variable_taus(img_4_points, tau1, tau2, tau3, tau4, default_threshold=500):

    tau_map = np.zeros((img_4_points.shape[0], img_4_points.shape[1]))

    for i in range(img_4_points.shape[0]):
        for j in range(img_4_points.shape[1]):
            # four quadrants
            if i < img_4_points.shape[0]//2 and j < img_4_points.shape[1]//2:
                if img_4_points[i,j] > default_threshold:
                    tau_map[i,j] = tau1
                else:
                    tau_map[i,j] = 10000

            elif i < img_4_points.shape[0]//2 and j >= img_4_points.shape[1]//2:
                if img_4_points[i,j] > default_threshold:
                    tau_map[i,j] = tau2
                else:
                    tau_map[i,j] = 10000

            elif i >= img_4_points.shape[0]//2 and j < img_4_points.shape[1]//2:
                if img_4_points[i,j] > default_threshold:
                    tau_map[i,j] = tau3
                else:
                    tau_map[i,j] = 10000

            else:
                if img_4_points[i,j] > default_threshold:
                    tau_map[i,j] = tau4
                else:
                    tau_map[i,j] = 10000

    return tau_map


#-----------------------------------------------Visualization functions-------------------------------------------------#

def show_image(image, vmin=100, vmax=4095, colorbar=False):
    '''
    Show an image with a fixed maximum and minimum value of the colormap

    Input:
        image : the image to show (numpy array)
        vmin : the minimum value of the colormap (float)
        vmax : the maximum value of the colormap (float)
        colorbar : boolean, if True, a colorbar is displayed (bool)
    '''
    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar()
    plt.axis('off')
    plt.show()

def update_frame(num, image_list, im, vmin, vmax):
    im.set_array(image_list[num])
    im.set_clim(vmin=vmin, vmax=vmax)  # Update color limits for each frame
    return im,

def show_simulation(list_simu, vmin=None, vmax=None):
    '''
    Show a simulation as an animation
    '''

    if type(list_simu) == list:
        list_simu = list_to_array(list_simu)

    if vmin==None or vmax==None:
        vmin = np.min(list_simu)
        vmax = np.max(list_simu)

    # Initialize the figure
    fig = plt.figure()

    # Initialize the image object
    # im = plt.imshow(list_simu[0], cmap='gray', vmin=vmin, vmax=vmax)
    # plt.close()

    im = plt.imshow(list_simu[0], cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.close()

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(list_simu), fargs=(list_simu, im, vmin, vmax), blit=True)

    # Convert the animation to HTML
    html_video = ani.to_jshtml()

    return HTML(html_video)

#-----------------------------------------------Photobleaching correction-------------------------------------------------#

# def correct_photobleaching_fixed_framework(denoised_images, tau_map): 

#     t = np.arange(0, len(denoised_images), 1).reshape(-1, 1, 1)
#     exponential_factor = np.exp(t/tau_map)
#     corrected_simulation = denoised_images*exponential_factor

#     return corrected_simulation

def correct_photobleaching_fixed_framework(denoised_images, tau_map): 
    # Create the time array
    t = np.arange(0, len(denoised_images), 1).reshape(-1, 1, 1)
    
    # Create a mask where tau_map is zero
    zero_tau_map_mask = (tau_map == 0)
    
    # Set the exponential factor to the max value where tau_map is zero
    tau_map = np.where(zero_tau_map_mask, 0.1, tau_map)

    # Calculate the exponential factor
    exponential_factor = np.exp(t / tau_map)
    
    # Apply the exponential factor to denoised images
    corrected_simulation = denoised_images * exponential_factor

    return corrected_simulation


def mean_background(y_pixel, x_pixel, _3Darray, plot=False):
    if type(_3Darray) == list:
        _3Darray = list_to_array(_3Darray)
    if plot:
        plt.plot(_3Darray[:,y_pixel, x_pixel])
    return np.mean(_3Darray[:,y_pixel, x_pixel])

#-----------------------------------------------Useful functions-------------------------------------------------#

def array_to_list(array_of_images):
    '''
    Convert a 3D numpy array to a list of 2D numpy arrays

    Input:
        array_of_images : the 3D numpy array (numpy array)

    Output:
        list_of_images : the list of 2D numpy arrays (list of numpy arrays)
    '''
    list_of_images = [array_of_images[i] for i in range(array_of_images.shape[0])]
    return list_of_images


def list_to_array(list_of_images):
    '''
    Convert a list of 2D numpy arrays to a 3D numpy array

    Input:
        list_of_images : the list of 2D numpy arrays (list of numpy arrays)

    Output:
        array_of_images : the 3D numpy array (numpy array)
    '''
    array_of_images = np.stack(list_of_images)
    return array_of_images


def clip_zeros(array):
    array[array < 0] = 0
    return array

def denoised_tensors_to_list_arrays(all_denoised_tensors):
    list_of_arrays = [tensor.detach().cpu().squeeze().numpy().astype(np.float64) for tensor in all_denoised_tensors]
    list_of_arrays = [clip_zeros(array) for array in list_of_arrays]
    return list_of_arrays


def get_var_name(var):
    for name, value in locals().items():
        if value is var:
            return name
        
def change_array_type(array, new_type):
    return array.astype(new_type)
        

def tau_amp_estimates(simulation_array, vmax_tau=None, vmax_amp=None, t0=0, time_step=1):
    '''
    Find the maximum estimated tau value for a given simulation array.
    '''
    if type(simulation_array) == list:
        simulation_array = list_to_array(simulation_array)

    t = np.arange(t0, simulation_array.shape[0], time_step)
    y = np.zeros(len(t))

    estimations_tau = np.zeros((simulation_array.shape[1], simulation_array.shape[2]))
    estimations_amp = np.zeros((simulation_array.shape[1], simulation_array.shape[2]))

    for i in range(simulation_array.shape[1]):
        for j in range(simulation_array.shape[2]):

            for time in range(len(t)):
                y[time] = simulation_array[time][i][j]

            estimated_amplitude = np.max(y)
            estimated_tau = -len(y)/np.log(y[-1]/y[0])
            estimations_tau[i][j] = estimated_tau
            estimations_amp[i][j] = estimated_amplitude

    if vmax_tau==None:
        vmax_tau=np.max(estimations_tau)

    estimations_tau[estimations_tau<0]=0

    plt.imshow(estimations_tau, vmin=0, vmax=vmax_tau)
    plt.colorbar()
    plt.show()



    if vmax_amp==None:
        vmax_amp=np.max(estimations_amp)

    estimations_amp[estimations_amp<0]=0
    plt.imshow(estimations_amp, vmin=0, vmax=vmax_amp)
    plt.colorbar()
    plt.show()
