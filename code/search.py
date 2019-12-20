class Search:
    def __init__(self, x, y):
        """
        :param x: starting x co-ordinate
        :param y: starting y co-ordinate
        """
        self.x = round(x,2)
        self.y = round(y,2)
        self.func = lambda x, y: (x**2) + (y**2)

    def get_neighbors(self, step = 0.1):
        """
        :param step: step size to take
        :return: a list of neighbors who are also of type Search
        """
        neighbors = []

        x = [step, step, 0, -step, -step, -step, 0, step]
        y = [0, step, step, step, 0, -step, -step, -step]
        for i in range(8):
            neighbors.append(self + Search(x[i], y[i]))
        return neighbors

    def score(self):
        """
        :return: the value of the function at self.x and self.y coordinates.
        """
        return self.func(self.x, self.y)

    def __add__(self, other):
        """
        a private function, need not be called from outside the class.
        overloads the + operator and returns the pairwise sum of respective co-ordinates.
        """
        return Search(self.x + other.x, self.y + other.y)

    def __hash__(self):
        return hash(str(self))

    def __getAsciiString(self):
        my_str = 'x = ' + str(self.x) + ' y = ' + str(self.y)
        return my_str

    def __str__(self):
        return self.__getAsciiString()