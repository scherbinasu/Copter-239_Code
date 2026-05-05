import math

class Point:
    x = 0
    y = 0

    def __init__(self, arg1, arg2=None):
        if arg2 is not None:
            arg1 = arg1, arg2
        self.x = arg1[0]
        self.y = arg1[1]
    def __str__(self):
        return 'Point('+str(self.x)+', '+str(self.y)+')'
    def to_int(self):
        return (int(self.x + 0.5), int(self.y + 0.5))

    def to_float(self):
        return (self.x, self.y)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def __abs__(self):
        return Point(abs(self.x), abs(self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y

    def __lt__(self, other):
        return self.x < other.x or self.y < other.y

    def __gt__(self, other):
        return self.x > other.x or self.y > other.y

    def __ge__(self, other):
        return self.x >= other.x or self.y >= other.y

    def __le__(self, other):
        return self.x <= other.x or self.y <= other.y

    def __iter__(self):
        yield iter((self.x, self.y))

    def __getitem__(self, item):
        return (self.x, self.x)[item]


class Vector:
    def __init__(self, arg1, arg2=None):
        if arg2 is not None or type(arg1) == tuple or type(arg1) == list:
            dy = arg2[0] - arg1[0]
            dx = arg2[1] - arg1[1]
            # print(f"dy={arg2[0]}-{arg1[0]} = {dy}, dx={arg2[1]}-{arg1[1]}={dx}")
            arg1 = math.atan2(dy, dx)+math.pi
            # print(180-math.degrees(arg1))
        self.radians = arg1
    def __str__(self):
        return 'Vector('+str(self.radians)+ ')'
    def get_degrees(self):
        return math.degrees(self.radians)

    def get_radians(self):
        return self.radians

    def point_to_vector_point(self, point_start: Point, dist):
        sin, cos = self.get_sin_cos()
        return Point(point_start[0]+sin*dist, point_start[1]+cos*dist)

    def get_sin_cos(self):
        return math.sin(self.radians), math.cos(self.radians)

    def __add__(self, other):
        return Vector((self.get_radians() + other.get_radians()) / 2)

    def perpendicular(self):
        return (Vector((self.radians + math.pi / 2 + math.pi) % (math.pi * 2)),
                Vector((self.radians + math.pi / 2) % (math.pi * 2)))

    def __reversed__(self):
        return Vector((self.radians + math.pi) % (math.pi * 2))

    def __neg__(self):
        return Vector((self.radians + math.pi) % (math.pi * 2))

    def __invert(self):
        return Vector((self.radians - math.pi) % (math.pi * 2))



class PID_regulator():
    _old_err = 0
    integral_err = 0

    def __init__(self, kp, ki, kd, setpoint=0):
        self.setpoint = setpoint
        self.kp = kp
        self.kd = kd
        self.ki = ki

    def __call__(self, input, setpoint=None, kp=None, ki=None, kd=None):
        if setpoint is None:
            setpoint = self.setpoint
        if kp is None:
            kp = self.kp
        if kd is None:
            kd = self.kd
        if ki is None:
            ki = self.ki

        err = setpoint - input
        i = self.integral_err
        self.integral_err += err * kp
        d = err - self._old_err
        self._old_err = err
        return err * kp + d * kd + i * ki
