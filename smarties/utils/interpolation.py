# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import numpy as np
from scipy.interpolate import griddata

from smarties.hn import Spin_maps

__all__ = [
    'perform_interpolation_scipy'
]

def perform_interpolation_scipy(
    dictionary_detector_to_interpolate: dict,
    dictionary_known_values: dict,
    slice_known_values: slice | None = None,
    slice_output: slice | None = None,
    method_interpolation: str = "linear",
    position_x_key: str = "wafer_x_mm",
    position_y_key: str = "wafer_y_mm",
    return_as_spin_maps: bool = True,
    stack_output_with_known_values: bool = False,
) -> dict | Spin_maps:
    """
    Interpolate values from known some detector positions on a focal plane and the known quantity values at those positions, 
    to the unknown quantity values related to some other positions on the focal plane, using scipy's griddata.

    Parameters
    ----------
    dictionary_detector_to_interpolate : dict
        Dictionary containing the detector positions position_x_key and position_y_key to interpolate to.
    dictionary_known_values : dict
        Dictionary containing the detector positions position_x_key and position_y_key related to the known values to interpolate from,
        as well as the known values themselves.
    slice_known_values : slice, optional
        Slice to apply to the known values before interpolation. Default is None (no slicing).
    slice_output : slice, optional
        Slice to apply to the output values after interpolation. Default is None (no slicing).
    method_interpolation : str, optional
        Interpolation method to use. Options are 'nearest', 'linear', or 'cubic. Default is 'linear'.
    position_x_key : str, optional
        Key in the dictionaries for the x positions. Default is 'wafer_x_mm'.
    position_y_key : str, optional
        Key in the dictionaries for the y positions. Default is 'wafer_y_mm'.
    return_as_spin_maps : bool, optional
        If True, return the results as a Spin_maps object. If False, return as a dictionary. Default is True.
    stack_output_with_known_values : bool, optional
        If True, stack the known values on top of the interpolated values in the output. Default is False.

    Returns
    -------
    dict or Spin_maps
        Interpolated values as a dictionary or Spin_maps object, depending on the return_as_spin_maps parameter.

    Notes
    -----
    If the chosen as linear or cubic leads to NaN values, the nearest interpolation will be used to (silently) fill those NaNs.

    WARNING: If stack_output_with_known_values is True, the output will have the known values stacked on top of the interpolated values.

    """

    assert position_x_key in dictionary_known_values, (
        f"{position_x_key} not found in dictionary_known_values"
    )
    assert position_y_key in dictionary_known_values, (
        f"{position_y_key} not found in dictionary_known_values"
    )
    assert position_x_key in dictionary_detector_to_interpolate, (
        f"{position_x_key} not found in dictionary_detector_to_interpolate"
    )
    assert position_y_key in dictionary_detector_to_interpolate, (
        f"{position_y_key} not found in dictionary_detector_to_interpolate"
    )

    assert method_interpolation in ["nearest", "linear", "cubic"], (
        f"method_interpolation must be one of 'nearest', 'linear', or 'cubic', got {method_interpolation}"
    )

    if slice_known_values is None:
        _slice = ...
    else:
        _slice = ..., slice_known_values

    if slice_output is None or np.any(np.array(slice_output) == np.array(...)):
        _slice_output = ...
    else:
        _slice_output = ..., slice_output

    keys_values_to_fit = list(dictionary_known_values.keys())
    keys_values_to_fit.remove(position_x_key)
    keys_values_to_fit.remove(position_y_key)

    results = dict() if not return_as_spin_maps else Spin_maps.from_dictionary(dict())

    for key in keys_values_to_fit:
        nearest_values = griddata(
            (
                dictionary_known_values[position_x_key],
                dictionary_known_values[position_y_key],
            ),
            dictionary_known_values[key][_slice],
            (
                dictionary_detector_to_interpolate[position_x_key],
                dictionary_detector_to_interpolate[position_y_key],
            ),
            method="nearest",
        )

        interpolated_values = griddata(
            (
                dictionary_known_values[position_x_key],
                dictionary_known_values[position_y_key],
            ),
            dictionary_known_values[key][_slice],
            (
                dictionary_detector_to_interpolate[position_x_key],
                dictionary_detector_to_interpolate[position_y_key],
            ),
            method=method_interpolation,
        )
        # Replace possible nans with nearest interpolation values
        interpolated_values[np.isnan(interpolated_values)] = nearest_values[
            np.isnan(interpolated_values)
        ]

        if not stack_output_with_known_values:
            results[key] = interpolated_values[_slice_output]
        else:
            results[key] = np.vstack(
                (
                    dictionary_known_values[key][_slice_output],
                    interpolated_values[_slice_output],
                )
            )
    return results
