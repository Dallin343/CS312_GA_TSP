import numpy as np


class State:
    def __init__(self, parent_matrix, parent_lower_bound, new_partial_path, new_unused_city_indices):
        self.matrix = parent_matrix
        self.partial_path = new_partial_path
        self.lower_bound = parent_lower_bound
        self.unused_city_indices = new_unused_city_indices

    def get_depth(self):
        return len(self.partial_path)

    def get_priority_value(self):
        if self.get_depth() == 0:
            return 0
        else:
            return self.lower_bound / (self.get_depth() * self.get_depth())
            # return self.lower_bound - self.get_depth() * 500

    # O(n^2)
    def reduce_matrix(self):

        # traverse each row O(n)
        for i in range(len(self.matrix)):

            # get the shortest edge in the row O(n)
            shortest_edge = np.inf
            for j in range(len(self.matrix)):
                shortest_edge = min(shortest_edge, self.matrix[i][j])

            # if the whole row isn't inf, add shortest edge to the lower bound and subtract it from each edge in row
            if shortest_edge != np.inf:
                self.lower_bound += shortest_edge
                # O(n)
                for j in range(len(self.matrix)):
                    self.matrix[i][j] -= shortest_edge

        # traverse each column O(n)
        for j in range(len(self.matrix)):

            # get the shortest edge in each column O(n)
            shortest_edge = np.inf
            for i in range(len(self.matrix)):
                shortest_edge = min(shortest_edge, self.matrix[i][j])

            # if the whole column isn't inf, add shortest edge to lower bound and subtract it from each edge in  column
            if shortest_edge != np.inf:
                self.lower_bound += shortest_edge
                # O(n)
                for i in range(len(self.matrix)):
                    self.matrix[i][j] -= shortest_edge

    # O(n)
    def infinitize(self, i, j):

        # add the cost of the chosen edge to the lower bound
        self.lower_bound += self.matrix[i][j]

        # make the row and column of edge all equal to infinity O(n)
        for k in range(len(self.matrix)):
            self.matrix[i][k] = np.inf
            self.matrix[k][j] = np.inf

        # make the symmetric edge infinity, as well
        self.matrix[j][i] = np.inf

    def __str__(self):

        output = '{'
        output += '\n\t\"partial_path\": ' + str(self.partial_path) + ', '
        output += '\n\t\"lower_bound\": ' + str(self.lower_bound) + ', '
        output += '\n\t\"unused_city_indices\": ' + str(self.unused_city_indices) + ', '
        output += '\n\t\"matrix\": '
        for row in self.matrix:
            output += '\n\t\t' + str(row)
        output += '\n}'

        return output

    def __repr__(self):
        return self.__str__()
