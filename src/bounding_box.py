class BoundingBox(object):
    """
    keeps coordinates of left upper corner and dimensions of bounding box
    """

    def __init__(self, string_list):
        """

        :param string_list: construct BoundingBox from list of strings containing coordinates of left upper corner
        and dimensions of bounding box
        """
        self.x0, self.y0, self.dx, self.dy = map(int, string_list)

    @staticmethod
    def get_bounding_boxes(file_path):
        """

        :param file_path: path to file containing data
        :return: dictionary of bounding boxes with key as image name
        """
        bounding_boxes = dict()
        with open(file_path) as file:
            for raw_line in file.readlines():
                tokens = raw_line.strip().split(' ')
                hash = tokens[0].replace("-", "")
                bounding_boxes[hash] = BoundingBox(tokens[1:])
        return bounding_boxes
