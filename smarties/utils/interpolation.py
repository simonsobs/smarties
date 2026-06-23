import numpy as np
from scipy.interpolate import griddata
from smarties.hn import Spin_maps

def perform_interpolation_scipy(
        dictionary_detector_to_interpolate,
        dictionary_known_values, 
        slice_known_values=None,
        slice_output=None,
        method_interpolation='linear',
        position_x_key='wafer_x_mm',
        position_y_key='wafer_y_mm',
        return_as_spin_maps=True,
        stack_output_with_known_values=False
    ):
    """
    WARNING: If stack_output_with_known_values is True, the output will have the known values stacked on top of the interpolated values. 
    """

    assert position_x_key in dictionary_known_values, f"{position_x_key} not found in dictionary_known_values"
    assert position_y_key in dictionary_known_values, f"{position_y_key} not found in dictionary_known_values"
    assert position_x_key in dictionary_detector_to_interpolate, f"{position_x_key} not found in dictionary_detector_to_interpolate"
    assert position_y_key in dictionary_detector_to_interpolate, f"{position_y_key} not found in dictionary_detector_to_interpolate"

    assert method_interpolation in ['nearest', 'linear', 'cubic'], f"method_interpolation must be one of 'nearest', 'linear', or 'cubic', got {method_interpolation}"

    if slice_known_values is None:
        _slice = ...
    else:
        _slice = ...,slice_known_values
    
    if slice_output is None or np.any(np.array(slice_output) == np.array(...)) :
        _slice_output = ...
    else:
        _slice_output = ...,slice_output

    keys_values_to_fit = list(dictionary_known_values.keys())
    keys_values_to_fit.remove(position_x_key)
    keys_values_to_fit.remove(position_y_key)
    
    
    results = dict() if not return_as_spin_maps else Spin_maps.from_dictionary(dict())
    


    for key in keys_values_to_fit:
        nearest_values = griddata( 
            (
                dictionary_known_values[position_x_key],
                dictionary_known_values[position_y_key]
            ), 
            dictionary_known_values[key][_slice], 
            (dictionary_detector_to_interpolate[position_x_key],
            dictionary_detector_to_interpolate[position_y_key]), 
            method='nearest'
        )

        interpolated_values = griddata((
                dictionary_known_values[position_x_key],
                dictionary_known_values[position_y_key]
            ), 
            dictionary_known_values[key][_slice], 
            (dictionary_detector_to_interpolate[position_x_key],
            dictionary_detector_to_interpolate[position_y_key]),
            method=method_interpolation
        )
        # Replace possible nans with nearest interpolation values
        interpolated_values[np.isnan(interpolated_values)] = nearest_values[np.isnan(interpolated_values)]
        
        if not stack_output_with_known_values:
            results[key] = interpolated_values[_slice_output]
        else:
            results[key] = np.vstack((dictionary_known_values[key][_slice_output], interpolated_values[_slice_output]))
    return results
