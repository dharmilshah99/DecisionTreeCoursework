class Node:
    """Represents a Decision Tree Node."""

    def __init__(self, left=None, right=None, attribute=None, value=None, label=None):
        """Initializes Decision Tree Node.

        Args:
            left (Node, optional): Node left of current node. Defaults to None.
            right (Node, optional): Node right of current node. Defaults to None.
            attribute (int, optional): Attribute to split dataset upon. Defaults to None.
            value (float, optional): Split point value. Defaults to None.
            label (int, optional): Classification label of Node. Defaults to None.
        """
        self.left = left
        self.right = right
        self.attribute = attribute
        self.value = value
        self.label = label

    def is_root(self):
        """Checks if Node is a root.

        Returns:
            True if Node has children. False, otherwise.
        """
        return (self.left == None) and (self.right == None)


if __name__ == "__main__":
    pass
