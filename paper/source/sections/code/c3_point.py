# This is the class definition. Object oriented programming has the concept
# of inheritance, whereby classes may be "children" of other classes. The
# parent class is specified in the parentheses. When defining a class with
# no parent, the base class `object` is specified instead.
class Point(object):

    # The __init__ function is a special method that is run whenever an
    # object is created. In this case, the initial coordinates are set to
    # the origin. `self` is a variable which refers to the object instance
    # itself.
    def __init__(self):
        self.x = 0
        self.y = 0

    def change_x(self, dx):
        self.x = self.x + dx

    def change_y(self, dy):
        self.y = self.y + dy

# An object of class Point is created
point_object = Point()

# The object exposes it's attributes
print(point_object.x)  # 0

# And we can call the object's methods
# Notice that although `self` is the first argument of the class method,
# it is automatically populated, and we need only specify the other
# argument, `dx`.
point_object.change_x(-2)
print(point_object.x)  # -2

# This is the new class definition. Here, the parent class, `Point`, is in
# the parentheses.
class Vector(Point):

    def __init__(self, x, y):
        # Call the `Point.__init__` method to initialize the coordinates
        # to the origin
        super(Vector, self).__init__()

        # Now change to coordinates to those provided as arguments, using
        # the methods defined in the parent class.
        self.change_x(x)
        self.change_y(y)

    def length(self):
        # Notice that in Python the exponentiation operator is a double
        # asterisk, "**"
        return (self.x**2 + self.y**2)**0.5

# An object of class Vector is created
vector_object = Vector(1, 1)
print(vector_object.length())  # 1.41421356237
