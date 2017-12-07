def radec2v_global(ra, dec):
    """ Convert from J2000 RA/DEC coords to global(J2000) vector

    Parameters
    ----------
    ra : list of floats 
        Actual J2000 RA coordinate of interest (units=degrees)'
    dec : list of floats
        Actual J2000 DEC coordinate of interest (units=degrees)'

    Return
    ------
    v_global: array of floating numbers
        Unit vector in J2000 'pointed' at ra/dec" 
            
    """

    import numpy as np

    v_global = np.empty((3, len(ra)))
    v_global[0, :] = np.cos(np.deg2rad(dec)) * np.cos(np.deg2rad(ra))
    v_global[1, :] = np.cos(np.deg2rad(dec)) * np.sin(np.deg2rad(ra))
    v_global[2, :] = np.sin(np.deg2rad(dec))

    return v_global

def radec_cam2v_cam(ra_cam, dec_cam):
    """
    Parameters
    ----------

    ra_cam : list/array of floats 
	'fake' RA angle in camera frame (scalar or array of angles)"
    dec_cam : list/array of floats 
	'fake' DEC angle in camera frame (scalar or array of angles)"

    Returns
    -------
    v_cam : array of floating numbers
	Numpy array

    """
    return radec2v_global(ra_cam, dec_cam)

def v_global2v_sc(v_global, image):
    """Convert global coordinate vector to spacecraft coordinate vector

    Parameters
    ----------
    v_global : array
        Vector in global (J2000) frame (can be 3-d  vector or array of vectors)'
    image : str
        Pick stellar image, (e.g. 'mx1' or 'my3')" 

    Returns
    -------
    v_sc : array
        Transformed unit vector in S/C frame '
    """
    from scipy.io import readsav
    from pyquaternion import Quaternion
    import numpy as np

    # Read .sav files
    q_global = readsav("./data/q_global.sav", python_dict=True)

    q_vec = q_global["q_global_"+image]
    q_vec = np.append(q_vec[-1], q_vec[0:-1])
    quat = Quaternion(q_vec)
    quat_invert = quat.conjugate

    v_sc = np.array([quat_invert.rotate(v_global[:, i]) for i in range(v_global.shape[1])])
    v_sc = v_sc.transpose()

    return v_sc

def v_sc2v_cam(v_sc, cam): 
    """ convert vector in spacecraft coordinate system to vector in camera coordinate system

    Parameters
    ----------
    v_sc : array 
        Single vector or array of vectors in S/C frame [3,:]
    cam : str
        Camera (e.g., 'px', 'mx' ,'py',or 'my')

    Returns
    -------
    v_cam : array
        Vector in camera reference frame 
    """

    from scipy.io import readsav
    from pyquaternion import Quaternion
    import numpy as np

    # Read .sav files
    q_sc = readsav("./data/q_sc.sav", python_dict=True)

    q_vec = q_sc['q_sc_'+cam]
    q_vec = np.append(q_vec[-1], q_vec[0:-1])

    quat = Quaternion(q_vec)
    #quat_invert = quat.inverse
    quat_invert = quat.conjugate
    v_cam = np.array([quat_invert.rotate(v_sc[:, i]) for i in range(v_sc.shape[1])])
    v_cam = v_cam.transpose()

    return v_cam


def v_cam2radec_cam(v_cam):

    """ convert vector in camera coordinate system to 'fake' ra/dec camera coords
	using spherical coord conversion:

    Parameters
    ----------
    v_cam : array
	Single vector or array of vectors in camera frame ie [3,:]

    Returns
    -------
    ra_cam : array
	'fake' RA angle in camera frame (scalar or array of angles)"
    dec_cam : array
	'fake' DEC angle in camera frame (scalar or array of angles)"
    """

    import numpy as np

    xyzn = np.array([v_cam[:, i] / np.linalg.norm(v_cam[:, i]) \
		     for i in range(v_cam.shape[1])])
    xyzn = xyzn.transpose()
    #dec_cam = 90 - np.rad2deg(np.arccos(xyzn[2, :]))
    dec_cam = np.rad2deg(np.arcsin(xyzn[2, :]))
    ra_cam = np.rad2deg(np.arctan2(xyzn[1,:], xyzn[0,:])) % 360

    return ra_cam, dec_cam

def radec_cam2pixel(cam, ra_cam, dec_cam, wpix=0.014, proj="1a"):
    """ Convert image coordinates (pixel locations) to 'fake' camera RA/DEC 
        using the specified camera model 

    Parameters
    ----------
    cam : str
        Enter camera with single quotes('px','mx','py',or 'my')
    ra_cam : array
        'fake' ra angle in camera frame
    dec_cam : array 
        'fake' dec angle in camera frame

    Returns
    -------
    x,y : float
        Pixel coordinates 
    """
    import numpy as np

    if cam == "px":
	flen = 26.227412
	if proj == '1a':
	    flen = 24.716602
	    print('1a px model')
	if proj == 'flen_offset':
	    flen = 24.716602 + flen_offset
	    print('1a px model flen_offset')
   
    if cam == 'mx':
	flen = 26.417139
	if proj == '1a':
	    flen = 24.878283
	    print('1a mx model')
	if proj == 'flen_offset':
	    flen = 24.878283 + flen_offset
	    print('1a px model flen_offset')

    if cam == "py":
	flen = 26.325814
	if proj == '1a':
	    flen = 24.800448
	    print('1a px model')
	if proj == 'flen_offset':
	    flen = 24.800448 + flen_offset
	    print('1a px model flen_offset')

    if cam == 'my':
	flen = 26.447683
	if proj == '1a':
	    flen = 24.904327
	    print('1a mx model')
	if proj == 'flen_offset':
	    flen = 24.904327 + flen_offset
	    print('1a px model flen_offset')
	
    theta = np.deg2rad(90 - dec_cam)
    phi = np.deg2rad(ra_cam)
    pixel_dist = np.array([np.tan(theta[i])*flen/wpix for i in range(len(ra_cam))])
    pixel_slope = np.array([np.tan(phi[i]) for i in range(len(ra_cam))])
    x = np.array([np.sqrt((pixel_dist[i]**2)/(1+pixel_slope[i]**2)) for i in range(len(ra_cam))])
    y = np.array([np.sqrt((pixel_dist[i]**2)/(1+(1./pixel_slope[i])**2)) for i in range(len(ra_cam))])

    # fix signs of x/y values

    v_cam = radec_cam2v_cam(ra_cam, dec_cam)
    # quadrant 2
    y = np.array([-y[i] if (v_cam[0,i] >= 0 and v_cam[1,i] < 0) else y[i] for i in range(len(ra_cam))])

    # quadrant 3
    x = np.array([-x[i] if (v_cam[0,i] < 0 and v_cam[1,i] < 0) else x[i] for i in range(len(ra_cam))])
    y = np.array([-y[i] if (v_cam[0,i] < 0 and v_cam[1,i] < 0) else y[i] for i in range(len(ra_cam))])

    # quadrant 4
    x = np.array([-x[i] if (v_cam[0,i] < 0 and v_cam[1,i] >= 0) else x[i] for i in range(len(ra_cam))])

    x = x + 760
    y = y + 760

    return x,y

def radec2pixel(ra, dec, image, cam, proj='1a'):

    import numpy as np

    v_global = radec2v_global(ra, dec)
    v_sc = v_global2v_sc(v_global, image)
    v_cam = v_sc2v_cam(v_sc, cam)
    ra_cam, dec_cam = v_cam2radec_cam(v_cam)
    x, y = radec_cam2pixel(cam, ra_cam, dec_cam)

    # cameras are at different orientations, so adjust pixels accordingly
    if cam == 'py':
	x2 = y
	y2 = x
	y2 = np.abs(1520.-y2)
	x = x2
	y = y2

    if cam == 'my':
	y2 = x
	x2 = y
	x2 = np.abs(1520.-x2)
	x = x2
	y = y2

    if cam == 'mx':
	x = np.abs(1520.-x)
	y = np.abs(1520.-y)

    output = {'v_global' : v_global, 
              'v_sc' : v_sc,
              'v_cam' : v_cam,
              'ra_cam' : ra_cam,
              'dec_cam' : dec_cam,
              'x' : x,
              'y' : y}
    return output

if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv("./data/image_database.txt", index_col=[0])
    ra = [220.617]
    dec = [-64.949]
    image = "mx3" 
    cam = "mx"
    proj='1a'
    output = radec2pixel(ra, dec, image, cam, proj='1a')

