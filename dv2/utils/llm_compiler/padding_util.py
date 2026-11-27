'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import numpy as np
from onnx import TensorProto

def next_multiple_of(N:int, n:int):
    if (N % n == 0):
        return N
    else:
        next_multiple = ((N + n - 1) // n) * n
        print(f"{N} aligned with {n} is {next_multiple}")
        return next_multiple

# Note for code readability. Adding type of arr_shape is skipped because
# 1. It is not a public API provided by this library. It is internal
# 2. Telling that type is google.protobuf.pyext._message.RepeatedScalarContainer
#    does not add any special value over already clear that it is a shape.
# We can definitely add it for stricter type check, but here it is not needed.
def pad_to_shape(arr_raw_bytes:np.ndarray, arr_shape, target_shape):
    """
    Pads a NumPy array with zeros to match target_shape.
    """
    arr = np.frombuffer(arr_raw_bytes, dtype=np.float32)
    arr = arr.reshape(arr_shape)

    if len(arr_shape) != len(target_shape):
        print(f"arr_shape:{arr_shape}  target_shape:{target_shape}")
        raise ValueError("Target shape must have same number of dimensions as array")

    pad_widths = []
    for current, target in zip(arr_shape, target_shape):
        if target < current:
            raise ValueError("Target shape must be >= current shape in all dimensions")
        pad_widths.append((0, target - current))  # pad at the end

    return np.pad(arr, pad_widths, mode='constant', constant_values=0)

def align(initializer:TensorProto, n:int):
    new_dims = []
    for i in range(len(initializer.dims)):
      new_dims.append(next_multiple_of(initializer.dims[i], n))
    print(f"new_dims: {new_dims}")
    initializer.raw_data = pad_to_shape(np.frombuffer(initializer.raw_data, dtype=np.float32), initializer.dims, new_dims).tobytes()
    initializer.dims[:] = new_dims
    return initializer

def assert_align_works(input:np.array, align_with:int, expected_output: np.array):
    initializer = TensorProto()
    initializer.name = 'my_tensor'
    initializer.data_type = TensorProto.FLOAT
    initializer.dims.extend(input.shape)       # repeated field
    initializer.raw_data = input.tobytes()     # binary data

    aligned_initializer = align(initializer, align_with)
    aligned_raw_weights = np.frombuffer(aligned_initializer.raw_data, dtype=np.float32)
    np.testing.assert_array_equal(
        # If there is a shape mismatch, the reshape will fail
        aligned_raw_weights.reshape(aligned_initializer.dims), expected_output)

# Few sample calls in order to:
# 1. Quick test to see if above methods are working fine.
# 2. Tell how the above methods are to be used.
if __name__ == "__main__":
    assert next_multiple_of(0, 64) == 0
    assert next_multiple_of(59, 64) == 64
    assert next_multiple_of(64, 64) == 64
    assert next_multiple_of(65, 64) == 128

    assert_align_works(
        np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], dtype=np.float32),
        2,
        np.array([[1.0, 2.0, 3.0, 0],
                  [4.0, 5.0, 6.0, 0]], dtype=np.float32)
    )

    assert_align_works(
        np.array([[1.0, 2.0, 3.0, 4.0],
                  [4.0, 5.0, 6.0, 7.0]], dtype=np.float32),
        4,
        np.array([[1.0, 2.0, 3.0, 4.0],
                  [4.0, 5.0, 6.0, 7.0],
                  [  0,   0,   0,   0],
                  [  0,   0,   0,   0]], dtype=np.float32)
    )




