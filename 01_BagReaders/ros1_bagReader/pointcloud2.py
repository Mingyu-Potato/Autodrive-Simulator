'''
from ros_numpy pointcloud2
'''

from sensor_msgs.msg import PointCloud2, PointField
import numpy as np

DUMMY_FIELD_PREFIX = '__'

type_mappings = [(PointField.INT8, np.dtype('int8')),
                 (PointField.UINT8, np.dtype('uint8')),
                 (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')),
                 (PointField.INT32, np.dtype('int32')),
                 (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')),
                 (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1,
                PointField.UINT8: 1,
                PointField.INT16: 2,
                PointField.UINT16: 2,
                PointField.INT32: 4,
                PointField.UINT32: 4,
                PointField.FLOAT32: 4,
                PointField.FLOAT64: 8}

def fields_to_dtype(fields, point_step):
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
            fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))