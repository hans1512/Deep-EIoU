class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def bottom_center(self):
        return (int((self.x1 + self.x2) // 2), int(self.y2))

    @property
    def width(self):
        return self.x2 - self.x1
