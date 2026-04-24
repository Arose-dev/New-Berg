MODULES_SIGNATURES = """
\"\"\"
Locates objects in an image. Object prompts should be simple and contain few words. To return all objects, pass "objects" as the prompt.

Args:
    image (image): Image to search.
    object_prompt (string): Description of object to locate.
Returns:
    list: A list of bounding boxes [xmin, ymin, xmax, ymax] for all of the objects located in pixel space.
\"\"\"
def loc(image, object_prompt):

\"\"\"
Answers a question about an object shown in a bounding box.

Args:
    image (image): Image of the scene.
    question (string): Question about the object in the bounding box.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, question, bbox):

\"\"\"
Returns the depth of an object (specified by a bounding box) in meters.

Args:
    image (image): Image of the scene.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.

Returns:
    float: The depth of the object specified by the coordinates in meters.
\"\"\"
def depth(image, bbox):

\"\"\"
Checks if two bounding boxes correspond to the same object.

Args:
    image (image): Image of the scene.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] containing object1.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] containing object2.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
\"\"\"
def same_object(image, bbox1, bbox2):

\"\"\"
Returns the width and height of the object in 2D pixel space.

Args:
    image (image): Image of the scene.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.

Returns:
    tuple: (width, height) of the object in 2D pixel space.
\"\"\"
def get_2D_object_size(image, bbox):

"""
