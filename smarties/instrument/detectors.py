from copy import deepcopy
import numpy as np
import yaml
import quaternionarray as qa



# WARNING: Not working
# def get_polarization_angles(detector_features, telescope='LAT'):
#     # TODO: Recheck because a priori wrong!! 
#     try:
#         import sotodlib.sim_hardware
#     except ImportError:
#         raise ImportError("sotodlib.sim_hardware is required to run this function. Please install the sotodlib package.")



#     hardware = sotodlib.sim_hardware.sim_nominal()
#     sotodlib.sim_hardware.sim_detectors_toast(hardware, telescope)

#     data_detectors = hardware.data['detectors']

#     zaxis = np.array([0, 0, 1], dtype=np.float64)
#     xaxis = np.array([1, 0, 0], dtype=np.float64)
    
#     array_polang = []
#     list_suffix = ['_A', '_B']

#     for d in detector_features:
#         for suffix in list_suffix:
#             quat = np.array(data_detectors[d+suffix]['quat']).astype(np.float64)
#             rdir = qa.rotate(quat, zaxis).flatten()
#             # ang = np.arctan2(rdir[1], rdir[0])
#             orient = qa.rotate(quat, xaxis).flatten()
#             polang = np.arctan2(orient[1], orient[0])
#             # mag = np.arccos(rdir[2]) * 180.0 / np.pi
#             # xpos.append(mag * np.cos(ang))
#             # ypos.append(mag * np.sin(ang))
#             # detectors.append(d)
#             # detpol.append(polang)
#             array_polang.append(polang)

#     return np.array(array_polang)


def get_polarization_angles(detector_features, telescope='LAT'):
    try:
        import sotodlib.sim_hardware
    except ImportError:
        raise ImportError("sotodlib.sim_hardware is required to run this function. Please install the sotodlib package.")

    hardware = sotodlib.sim_hardware.sim_nominal()
    sotodlib.sim_hardware.sim_detectors_toast(hardware, telescope)

    data_detectors = hardware.data['detectors']

    return np.array([np.deg2rad(data_detectors[det]['pol_ang']) for det in detector_features])

def save_extended_final_maps(
        final_spin_maps, 
        mask_mpi_from_total_mask, 
        nstokes,
        nside,
        path_output):

    extended_final_maps = np.zeros((nstokes,12*nside**2), dtype=complex)
    if nstokes == 3 or nstokes == 1:
        extended_final_maps[0, mask_mpi_from_total_mask != 0] = final_spin_maps[0]
    if nstokes == 3 or nstokes == 2:
        final_Q_map = (final_spin_maps[-2] + final_spin_maps[2])/2.
        final_U_map = 1j*(final_spin_maps[-2] - final_spin_maps[2])/2.

        extended_final_maps[-2, mask_mpi_from_total_mask != 0] = final_Q_map.real
        extended_final_maps[-1, mask_mpi_from_total_mask != 0] = final_U_map.real

    print("Saving map into", path_output)
    np.save(path_output, extended_final_maps[:,mask_mpi_from_total_mask!=0])

def get_ellipticities_values_from_yaml_file(
        detector_features_all, 
        sigma_FWHM,  # arcmin
        path_values,
        ellipticity_parameter_convention='Third flattening',
        input_type='ellipticity_parameters'
    ):
    assert input_type in ['ellipticity_parameters', 'dp_dc', 'delta_sigma_angle'], "input_type must be either 'ellipticity_parameters' or 'dp_dc' or 'delta_sigma_angle'"
    assert ellipticity_parameter_convention in ['Third flattening', 'Third eccentricity'], "ellipticity_parameter_convention must be either 'Third flattening' or 'Third eccentricity'"

    with open(path_values) as file:
        ellipticity_file = yaml.safe_load(file)

    sigma_cs = np.asarray(sigma_FWHM) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)

    if input_type == 'dp_dc' or input_type == 'delta_sigma_angle':
        if input_type == 'dp_dc':
            delta_sigma = np.sqrt([
                    ellipticity_file[det_name]['dc']**2 + ellipticity_file[det_name]['dp']**2
                    for det_name in detector_features_all
                    ]) * sigma_cs / 2. 

            ellipticity_angle = np.array([
                np.arctan2(-ellipticity_file[det_name]['dc'], ellipticity_file[det_name]['dp']) /4. 
                for det_name in (detector_features_all)
                ])
        elif input_type == 'delta_sigma_angle':
            delta_sigma = np.array([
                ellipticity_file[det_name]['delta_sigma'] 
                for det_name in detector_features_all
                ]) * sigma_cs

            ellipticity_angle = np.array([
                np.deg2rad(ellipticity_file[det_name]['angle']) 
                for det_name in (detector_features_all)
                ])

        if ellipticity_parameter_convention == 'Third flattening':
            # Third flattening
            # f = (a-b)/(a+b) = (sigma_maj - sigma_min)/(sigma_maj + sigma_min)
            ellipticity_parameter = delta_sigma / sigma_cs
        elif ellipticity_parameter_convention == 'Third eccentricity':
            ellipticity_parameter = (
                (sigma_cs + delta_sigma)**2 - (sigma_cs - delta_sigma)**2
                ) / (
                    (sigma_cs + delta_sigma)**2 + (sigma_cs - delta_sigma)**2
                )
    elif input_type == 'ellipticity_parameters':
        ellipticity_parameter = np.array([
            ellipticity_file[det_name]['ellipticity_value'] 
            for det_name in detector_features_all
            ])
        ellipticity_angle = np.array([
            np.deg2rad(ellipticity_file[det_name]['ellipticity_angle']) 
            for det_name in (detector_features_all)
            ])
    
    return ellipticity_parameter, ellipticity_angle

def get_detector_names_from_yaml_file(
        path_yaml
    ):

    with open(path_yaml) as file:
        dictionary_detector = yaml.safe_load(file)
        
    detector_features = []

    def fill_pixels_interval(pixel_name):
        if ':' in pixel_name:
            first_last_pixels = pixel_name.split(':')
            return ['p' + str(i).zfill(3) for i in range(
                int(first_last_pixels[0].split('p')[-1]),
                int(first_last_pixels[1].split('p')[-1]) + 1
                )
            ]
        else:
            return [pixel_name]

    # The same pixels are assumed for all wafers 
    default_pixels = dictionary_detector['detector_wafers']['default']
    for wafer, pixels in dictionary_detector['detector_wafers'].items():
        if wafer == 'default':
            continue
        if 'default' in pixels:
            pixel_distribution_chosen = default_pixels
        else:
            pixel_distribution_chosen = pixels

        if type(pixel_distribution_chosen) is str and ':' not in pixel_distribution_chosen:
            pixel_distribution_chosen = [pixel_distribution_chosen]
        if ':' in pixel_distribution_chosen:
            pixel_distribution_chosen = fill_pixels_interval(pixel_distribution_chosen)
        
        count_size = 0
        for j, pixel in enumerate(deepcopy(pixel_distribution_chosen)):
            if ':' in pixel:
                filled_pixels = fill_pixels_interval(pixel)
                pixel_distribution_chosen = pixel_distribution_chosen[:j + count_size] + filled_pixels + pixel_distribution_chosen[j+1+count_size:]
                count_size += len(filled_pixels) - 1
            
        for pixel in pixel_distribution_chosen:
            for suffix in dictionary_detector['h_n']['list_suffix']:
                str_detector_h_n = wafer + '_' + pixel + suffix
                detector_features.append(str_detector_h_n)
    return detector_features
