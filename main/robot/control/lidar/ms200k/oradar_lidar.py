import asyncio
import ctypes
import numpy as np
import time
import os

_LIB_PATH = os.path.join(os.path.dirname(__file__), 'liboradar_sdk.so')
_lib = ctypes.CDLL(_LIB_PATH)

class PointData(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_ushort),
        ("intensity", ctypes.c_ushort),
        ("angle", ctypes.c_float)
    ]

class FullScanData(ctypes.Structure):
    _fields_ = [
        ("data", PointData * 4097),
        ("vailtidy_point_num", ctypes.c_ushort),
        ("speed", ctypes.c_double)
    ]

class ORDLidar(ctypes.Structure):
    _fields_ = [("lidar", ctypes.c_void_p)]

# Прототипы
_lib.oradar_lidar_create.argtypes = [ctypes.c_uint8, ctypes.c_int]
_lib.oradar_lidar_create.restype = ctypes.POINTER(ORDLidar)

_lib.oradar_lidar_destroy.argtypes = [ctypes.POINTER(ctypes.POINTER(ORDLidar))]
_lib.oradar_lidar_destroy.restype = None

_lib.oradar_set_serial_port.argtypes = [ctypes.POINTER(ORDLidar), ctypes.c_char_p, ctypes.c_int]
_lib.oradar_set_serial_port.restype = ctypes.c_bool

_lib.oradar_connect.argtypes = [ctypes.POINTER(ORDLidar)]
_lib.oradar_connect.restype = ctypes.c_bool

_lib.oradar_disconnect.argtypes = [ctypes.POINTER(ORDLidar)]
_lib.oradar_disconnect.restype = ctypes.c_bool

_lib.oradar_activate.argtypes = [ctypes.POINTER(ORDLidar)]
_lib.oradar_activate.restype = ctypes.c_bool

_lib.oradar_deactive.argtypes = [ctypes.POINTER(ORDLidar)]
_lib.oradar_deactive.restype = ctypes.c_bool

_lib.oradar_get_grabfullscan_blocking.argtypes = [
    ctypes.POINTER(ORDLidar),
    ctypes.POINTER(FullScanData),
    ctypes.c_int
]
_lib.oradar_get_grabfullscan_blocking.restype = ctypes.c_bool

class LidarError(Exception):
    pass

class LidarReader:
    def __init__(self, port='/dev/ttyUSB0', baudrate=230400, timeout_ms=3000):
        self.port = port.encode('utf-8')
        self.baudrate = baudrate
        self.timeout_ms = timeout_ms
        self._lidar_ptr = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._lidar_ptr = _lib.oradar_lidar_create(0, 1)
        if not self._lidar_ptr:
            raise LidarError("Не удалось создать экземпляр драйвера")
        if not _lib.oradar_set_serial_port(self._lidar_ptr, self.port, self.baudrate):
            self._cleanup()
            raise LidarError("Ошибка установки параметров порта")
        if not _lib.oradar_connect(self._lidar_ptr):
            self._cleanup()
            raise LidarError("Ошибка подключения к лидару")
        time.sleep(0.5)
        if not _lib.oradar_activate(self._lidar_ptr):
            self._cleanup()
            raise LidarError("Ошибка активации лидара")
        time.sleep(2)
        self._started = True

    def get_scan(self):
        if not self._started:
            raise LidarError("Лидар не запущен. Вызовите start()")
        scan_data = FullScanData()
        ret = _lib.oradar_get_grabfullscan_blocking(
            self._lidar_ptr,
            ctypes.byref(scan_data),
            self.timeout_ms
        )
        if not ret:
            return None
        num = scan_data.vailtidy_point_num
        dtype = np.dtype([('angle', 'f4'), ('distance', 'f4'), ('intensity', 'f4')])
        arr = np.zeros(num, dtype=dtype)
        for i in range(num):
            p = scan_data.data[i]
            arr[i]['angle'] = (p.angle+0)%360
            arr[i]['distance'] = p.distance / 1000.0
            arr[i]['intensity'] = p.intensity
        return arr

    async def get_scan_async(self):
        if not self._started:
            raise LidarError("Лидар не запущен. Вызовите start()")
        scan_data = FullScanData()
        ret = await asyncio.to_thread(_lib.oradar_get_grabfullscan_blocking,
            self._lidar_ptr,
            ctypes.byref(scan_data),
            self.timeout_ms
        )
        if not ret:
            return None
        num = scan_data.vailtidy_point_num
        dtype = np.dtype([('angle', 'f4'), ('distance', 'f4'), ('intensity', 'f4')])
        arr = np.zeros(num, dtype=dtype)
        for i in range(num):
            p = scan_data.data[i]
            arr[i]['angle'] = p.angle
            arr[i]['distance'] = p.distance / 1000.0
            arr[i]['intensity'] = p.intensity
        return arr

    def stop(self):
        if self._lidar_ptr:
            _lib.oradar_lidar_destroy(ctypes.byref(self._lidar_ptr))
            self._lidar_ptr = None
        self._started = False

    def _cleanup(self):
        if self._lidar_ptr:
            _lib.oradar_lidar_destroy(ctypes.byref(self._lidar_ptr))
            self._lidar_ptr = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
