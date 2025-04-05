import numpy as np

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    
    Parameters:
    -----------
    image : 2D array
        The 2D image
    center : tuple
        The [x,y] pixel coordinates used as the center. The default is 
        None, which then uses the center of the image (including 
        fractional pixels).
    
    Returns:
    --------
    radialprofile : 1D array
        The radial profile.
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    
    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
    
    # Calculate the radius from the center
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Get the integer part of the radii
    r_int = r.astype(int)
    
    # Create a list of all radii in the original image
    r_max = np.max(r_int)
    
    # Create empty 1D arrays to store values
    tbin = np.bincount(r_int.ravel(), image.ravel())
    nr = np.bincount(r_int.ravel())
    
    # Divide the sum of the values at each radius by the count of pixels at that radius
    # Avoid division by zero
    nr = np.maximum(nr, 1)  # Replace zeros with ones
    radialprofile = tbin / nr
    
    return radialprofile[:r_max+1] 