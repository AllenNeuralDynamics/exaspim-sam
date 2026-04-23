import numpy as np


def resize_dask(image, scale_factor, order=1, output_chunks=(128, 256, 256)):
    """
    Resize a 3D Dask array using an affine transformation.

    This function scales a 3D Dask array by the given scale_factor along each dimension.
    It uses an affine transformation where the matrix maps output coordinates to input
    coordinates. For example, if scale_factor is 2, the output array will be twice as large
    in each dimension. When working with binary masks, consider using order=0 to preserve
    the binary nature.

    Parameters
    ----------
    image : dask.array
        The input 3D Dask array to be resized.
    scale_factor : float
        The scaling factor for each axis. For example, 2.0 will double the size.
    order : int, optional
        The order of the interpolation. Use order=0 for nearest-neighbor (good for binary
        masks), or higher orders for smoother results. Default is 1 (linear interpolation).
    output_chunks : tuple, optional
        The desired chunk size for the output Dask array. Default is (256, 256, 256).

    Returns
    -------
    dask.array
        The resized Dask array.
    """
    from dask_image.ndinterp import affine_transform

    # Construct a 4x4 homogeneous affine transformation matrix.
    # The matrix maps output coordinates into input coordinates.
    # Scaling factors are inverted because of this coordinate mapping.
    matrix = np.array([
        [1/scale_factor, 0, 0, 0],
        [0, 1/scale_factor, 0, 0],
        [0, 0, 1/scale_factor, 0],
        [0, 0, 0, 1]
    ])

    # Calculate the new output shape (assumes image has at least 3 dimensions).
    new_shape = tuple(int(dim * scale_factor) for dim in image.shape[:3])

    # Apply the affine transformation.
    resized_image = affine_transform(
        image,
        matrix=matrix,
        order=order,
        output_shape=new_shape,
        output_chunks=output_chunks
    )

    return resized_image
