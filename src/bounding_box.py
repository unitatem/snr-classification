class BoundingBox(object):
    """
    keeps coordinates of left upper corner and dimensions of bounding box
    """

    def __init__(self, string_list):
        """

        :param string_list: construct BoundingBox from list of strings containing coordinates of left upper corner
        and dimensions of boudning box
        """
        self.x0, self.y0, self.dx, self.dy = map(int, string_list)