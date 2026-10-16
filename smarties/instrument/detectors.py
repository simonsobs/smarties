# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import os
from copy import deepcopy
import logging

import numpy as np
import yaml

from smarties.hn import Spin_maps

logger = logging.getLogger(__name__)

__all__ = [
    "get_polarization_angles",
    "get_quantities_from_sotodlib",
    "save_spin_maps",
    "get_ellipticities_values_from_yaml_file",
    "get_detector_names_from_dictionary_detector",
    "get_detector_names_from_yaml_file",
    "get_list_detectors_from_yaml_and_h_n",
]



def get_polarization_angles(
    detector_features: list[str],
    telescope: str = "LAT",
) -> np.ndarray:
    """
    Get the polarization angles for a list of detectors using sotodlib.

    Parameters
    ----------
    detector_features : list
        List of detector names for which to retrieve polarization angles.
    telescope : str, optional
        The telescope name. Supported values are 'SAT1', 'SAT2', 'SAT3', and 'LAT'. Default is 'LAT'.
    
    Returns
    -------
    polarization_angles : np.ndarray
        Array of polarization angles in radians for the specified detectors.

    Note
    ----
    This function requires the `sotodlib` package to be installed. If it is not installed, an ImportError will be raised.
    Addditionally, the detector_features must matches the sotodlib detector names for the specified telescope. 
    """
    if telescope in ["SAT1", "SAT2", "SAT3", "LAT"]:
        try:
            import sotodlib.sim_hardware
        except ImportError:
            raise ImportError(
                "sotodlib.sim_hardware is required to run this function. Please install the sotodlib package."
            )

        hardware = sotodlib.sim_hardware.sim_nominal()
        sotodlib.sim_hardware.sim_detectors_toast(hardware, telescope)

        data_detectors = hardware.data["detectors"]

        return np.array(
            [np.deg2rad(data_detectors[det]["pol_ang"]) for det in detector_features]
        )
    else:
        raise ValueError(
            f"Telescope {telescope} not recognized. Supported telescopes are 'SAT1', 'SAT2', 'SAT3', and 'LAT'."
        )


def get_quantities_from_sotodlib(
    detector_features: list[str],
    list_keys_to_get: list[str],
    telescope: str = "LAT",
) -> dict[str, np.ndarray]:
    """
    Get the specified quantities for a list of detectors using sotodlib.

    Parameters
    ----------
    detector_features : list of str
        List of detector names for which to retrieve polarization angles.
    list_keys_to_get : list of str
        List of keys corresponding to the quantities to retrieve from sotodlib.
        Examples of valid keys include 'pol_ang', 'wafer_x_mm', 'wafer_y_mm', etc.
    telescope : str, optional
        The telescope name. Supported values are 'SAT1', 'SAT2', 'SAT3', and 'LAT'. Default is 'LAT'.
    
    Returns
    -------
    polarization_angles : np.ndarray
        Array of polarization angles in radians for the specified detectors.

    Note
    ----
    This function requires the `sotodlib` package to be installed. If it is not installed, an ImportError will be raised.
    Addditionally, the detector_features must matches the sotodlib detector names for the specified telescope. 
    The list_keys_to_get must be valid keys in the sotodlib detector data for the specified telescope.
    """
    if telescope in ["SAT1", "SAT2", "SAT3", "LAT"]:
        try:
            import sotodlib.sim_hardware
        except ImportError:
            raise ImportError(
                "sotodlib.sim_hardware is required to run this function. Please install the sotodlib package."
            )

        hardware = sotodlib.sim_hardware.sim_nominal()
        sotodlib.sim_hardware.sim_detectors_toast(hardware, telescope)

        data_detectors = hardware.data["detectors"]

        for key in list_keys_to_get:
            if key not in data_detectors[detector_features[0]]:
                raise ValueError(
                    f"Key {key} not found in the detector data. Available keys are: {list(data_detectors[detector_features[0]].keys())}"
                )

        return {
            key: np.array([data_detectors[det][key] for det in detector_features])
            for key in list_keys_to_get
        }
    else:
        raise ValueError(
            f"Telescope {telescope} not recognized. Supported telescopes are 'SAT1', 'SAT2', 'SAT3', and 'LAT'."
        )


def save_spin_maps(
    spin_maps: Spin_maps, 
    mask: np.ndarray, 
    nstokes: int, 
    nside: int, 
    path_output: str, 
    dtype: type | np.dtype = complex
) -> None:
    """
    Save the final spin maps into a file, assuming the corresponding pixels are 
    not given full sky coverage, and extending them to the full sky using the provided mask. 
    Only works for a healpix map with nside parameter.
    
    Parameters
    ----------
    spin_maps : Spin_maps
        Spin_maps object containing the spin maps to be saved.
    mask : np.ndarray
        Mask indicating the valid pixels in the spin maps. Non-zero values indicate valid pixels.
    nstokes : int
        The number of Stokes parameters.
    nside : int
        The nside parameter for the Healpix map.
    path_output : str
        The path to the output file.
    dtype : type | np.dtype, optional
        The data type of the output array. Default is complex.
    """
    assert nstokes in [1, 2, 3], (
        "The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)"
    )
    extended_maps = np.zeros((nstokes, 12 * nside**2), dtype=dtype)
    if nstokes == 3 or nstokes == 1:
        extended_maps[0, mask != 0] = spin_maps[0]
    if nstokes == 3 or nstokes == 2:
        extended_maps[-2, mask != 0] = ((spin_maps[-2] + spin_maps[2]) / 2.0).real
        extended_maps[-1, mask != 0] = (1j * (spin_maps[-2] - spin_maps[2]) / 2.0).real

    logger.info(f"Saving map into {path_output}")
    np.save(path_output, extended_maps[:, mask != 0])


def get_ellipticities_values_from_yaml_file(
    detector_features_all: list[str],
    sigma_FWHM: float,  # arcmin
    path_values: str,
    ellipticity_parameter_convention: str = "Third flattening",
    input_type: str = "ellipticity_parameters",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the ellipticity values and angles for a list of detectors from a YAML file.

    Parameters
    ----------
    detector_features_all : list of str
        List of detector names for which to retrieve ellipticity values and angles.
    sigma_FWHM : float
        Full width at half maximum (FWHM) of the beam in arcminutes.
    path_values : str
        Path to the YAML file containing ellipticity values and angles.
    ellipticity_parameter_convention : str, optional
        Convention for the output ellipticity parameter. 
        Supported values are given in smarties.tools.list_conventions
    input_type : str, optional
        Type of input values in the YAML file. Supported values are:
        - "ellipticity_parameters": The YAML file contains ellipticity values and angles.
        - "dp_dc": The YAML file contains plus and cross parameter convention.
        - "delta_sigma_angle": The YAML file contains delta sigma (the difference 
        between the major and minor axes divided by 2) and angles.
    
    Returns
    -------
    tuple_output : tuple of np.ndarray
        A tuple containing two arrays:
        - ellipticity_parameter: Array of ellipticity values for the specified detectors.
        - ellipticity_angle: Array of ellipticity angles (in radians) for the specified detectors

    Note
    ----
    The output ellipticity values and angles will be converted from the ellipticity_parameter_convention
    specified in the yaml file to the ellipticity_parameter_convention provided in the function arguments,
    if the input type is "dp_dc" or "delta_sigma_angle".      
    """
    assert input_type in ["ellipticity_parameters", "dp_dc", "delta_sigma_angle"], (
        "input_type must be either 'ellipticity_parameters' or 'dp_dc' or 'delta_sigma_angle'"
    )
    assert ellipticity_parameter_convention in [
        "Third flattening",
        "Third eccentricity",
    ], (
        "ellipticity_parameter_convention must be either 'Third flattening' or 'Third eccentricity'"
    )

    with open(path_values) as file:
        ellipticity_file = yaml.safe_load(file)

    sigma_cs = np.asarray(sigma_FWHM) / ((8 * np.log(2)) ** 0.5) * np.pi / (180 * 60)

    if input_type == "dp_dc" or input_type == "delta_sigma_angle":
        if input_type == "dp_dc":
            delta_sigma = (
                np.sqrt(
                    [
                        ellipticity_file[det_name]["dc"] ** 2
                        + ellipticity_file[det_name]["dp"] ** 2
                        for det_name in detector_features_all
                    ]
                )
                * sigma_cs
                / 2.0
            )

            ellipticity_angle = np.array(
                [
                    np.arctan2(
                        -ellipticity_file[det_name]["dc"],
                        ellipticity_file[det_name]["dp"],
                    )
                    / 4.0
                    for det_name in (detector_features_all)
                ]
            )
        elif input_type == "delta_sigma_angle":
            delta_sigma = (
                np.array(
                    [
                        ellipticity_file[det_name]["delta_sigma"]
                        for det_name in detector_features_all
                    ]
                )
                * sigma_cs
            )

            ellipticity_angle = np.array(
                [
                    np.deg2rad(ellipticity_file[det_name]["angle"])
                    for det_name in (detector_features_all)
                ]
            )

        if ellipticity_parameter_convention == "Third flattening":
            # Third flattening
            # f = (a-b)/(a+b) = (sigma_maj - sigma_min)/(sigma_maj + sigma_min)
            ellipticity_parameter = delta_sigma / sigma_cs
        elif ellipticity_parameter_convention == "Third eccentricity":
            ellipticity_parameter = (
                (sigma_cs + delta_sigma) ** 2 - (sigma_cs - delta_sigma) ** 2
            ) / ((sigma_cs + delta_sigma) ** 2 + (sigma_cs - delta_sigma) ** 2)
    elif input_type == "ellipticity_parameters":
        ellipticity_parameter = np.array(
            [
                ellipticity_file[det_name]["ellipticity_value"]
                for det_name in detector_features_all
            ]
        )
        ellipticity_angle = np.array(
            [
                np.deg2rad(ellipticity_file[det_name]["ellipticity_angle"])
                for det_name in (detector_features_all)
            ]
        )

    return ellipticity_parameter, ellipticity_angle


def get_detector_names_from_dictionary_detector(
    dictionary_detector: dict
) -> list[str]:
    """
    Get the detector names as ordered by sotodlib from a dictionary containing detector information. 

    Parameters
    ----------
    dictionary_detector : dict
        Dictionary containing detector information, including detector wafers, detector pixels and suffixes.
    
    Returns
    -------
    detector_features : list of str
        List of detector names with the sotodlib convention, constructed from the detector wafers, pixels and suffixes in the input dictionary.
    """

    detector_features = []

    def fill_pixels_interval(pixel_name):
        if ":" in pixel_name:
            first_last_pixels = pixel_name.split(":")
            return [
                "p" + str(i).zfill(3)
                for i in range(
                    int(first_last_pixels[0].split("p")[-1]),
                    int(first_last_pixels[1].split("p")[-1]) + 1,
                )
            ]
        else:
            return [pixel_name]

    # The same pixels are assumed for all wafers
    default_pixels = dictionary_detector["detector_wafers"]["default"]
    for wafer, pixels in dictionary_detector["detector_wafers"].items():
        if wafer == "default":
            continue
        if "default" in pixels:
            pixel_distribution_chosen = default_pixels
        else:
            pixel_distribution_chosen = pixels

        if (
            type(pixel_distribution_chosen) is str
            and ":" not in pixel_distribution_chosen
        ):
            pixel_distribution_chosen = [pixel_distribution_chosen]
        if ":" in pixel_distribution_chosen:
            pixel_distribution_chosen = fill_pixels_interval(pixel_distribution_chosen)

        count_size = 0
        for j, pixel in enumerate(deepcopy(pixel_distribution_chosen)):
            if ":" in pixel:
                filled_pixels = fill_pixels_interval(pixel)
                pixel_distribution_chosen = (
                    pixel_distribution_chosen[: j + count_size]
                    + filled_pixels
                    + pixel_distribution_chosen[j + 1 + count_size :]
                )
                count_size += len(filled_pixels) - 1

        for pixel in pixel_distribution_chosen:
            for suffix in dictionary_detector["h_n"]["list_suffix"]:
                str_detector_h_n = wafer + "_" + pixel + suffix
                detector_features.append(str_detector_h_n)
    return detector_features


def get_detector_names_from_yaml_file(
    path_yaml: str
) -> list[str]:
    """
    Get the detector names as ordered by sotodlib from a YAML file containing detector information.
    """
    with open(path_yaml) as file:
        dictionary_detector = yaml.safe_load(file)
    return get_detector_names_from_dictionary_detector(dictionary_detector)


def get_list_detectors_from_yaml_and_h_n(
    path_yaml_detectors: str, directory_h_maps: str
) -> list[str]:
    """
    Get the list of detectors from a YAML file and a directory containing h_n files.

    Parameters
    ----------
    path_yaml_detectors : str
        Path to the YAML file containing detector information as well as the prefix for the h_n files.
    directory_h_maps : str
        Path to the directory containing h_n files for the detectors.
    """
    with open(path_yaml_detectors) as file:
        dictionary_detectors = yaml.safe_load(file)

    detector_features = get_detector_names_from_dictionary_detector(
        dictionary_detectors
    )
    path_h_n = directory_h_maps + dictionary_detectors["h_n"]["file_prefix"][0]

    h_n_names_npz = []
    list_detector_to_remove = []

    _number_detectors = len(detector_features)
    for i in range(_number_detectors):
        if os.path.exists(
            path_h_n
            + detector_features[i]
            + "_"
            + dictionary_detectors["h_n"]["list_prefix"][0].replace('_', '')
            + "_cos_1."
            + dictionary_detectors["h_n"]["format_h_n"]
        ):
            h_n_names_npz.append(path_h_n + detector_features[i])
        else:
            list_detector_to_remove.append(detector_features[i])

    for det_features in list_detector_to_remove:
        detector_features.remove(det_features)
    return detector_features
