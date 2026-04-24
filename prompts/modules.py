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

MODULES_SIGNATURES_LEGO = """
\"\"\"
Locates objects in a LEGO assembly image. Object prompts should describe LEGO bricks or components.

Args:
    image (image): Image of the LEGO assembly to search.
    object_prompt (string): Description of LEGO object to locate. Examples: "bricks", "red bricks", "top brick".
Returns:
    list: A list of bounding boxes [xmin, ymin, xmax, ymax] for all of the objects located in pixel space.
\"\"\"
def loc(image, object_prompt):

\"\"\"
Answers a question about a LEGO brick or component shown in a bounding box.

Args:
    image (image): Image of the LEGO assembly.
    question (string): Question about the LEGO brick in the bounding box.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the LEGO brick.

Returns:
    string: Answer to the question about the LEGO brick in the image.
\"\"\"
def vqa(image, question, bbox):

\"\"\"
Checks if two bounding boxes correspond to the same LEGO brick.

Args:
    image (image): Image of the LEGO assembly.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] containing brick 1.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] containing brick 2.

Returns:
    bool: True if brick 1 is the same brick as brick 2, False otherwise.
\"\"\"
def same_object(image, bbox1, bbox2):

\"\"\"
Returns the depth of a LEGO object (specified by a bounding box) in the scene.

Args:
    image (image): Image of the LEGO assembly.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the LEGO object.

Returns:
    float: The depth of the LEGO object in the scene.
\"\"\"
def depth(image, bbox):

\"\"\"
Compares the 3D heights of two LEGO objects in the scene.
Uses depth estimation and image position to determine which object is higher.

Args:
    image (image): Image of the LEGO assembly.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] for the first LEGO object.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] for the second LEGO object.

Returns:
    string: 'first' if the first object is higher, 'second' if the second object is higher, 'same' if they are approximately the same height.
\"\"\"
def compare_3D_heights(image, bbox1, bbox2):

\"\"\"
Returns the width and height of a LEGO brick in 2D pixel space.

Args:
    image (image): Image of the LEGO assembly.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the LEGO brick.

Returns:
    tuple: (width, height) of the LEGO brick in 2D pixel space.
\"\"\"
def get_2D_object_size(image, bbox):
"""

MODULES_SIGNATURES_CLEVR = """
\"\"\"
Locates objects in an image. Object prompts should be 1 WORD MAX.

Args:
    image (image): Image to search.
    object_prompt (string): Description of object to locate. Examples: "spheres", "objects".
Returns:
    list: A list of x,y coordinates for all of the objects located in pixel space.
\"\"\"
def loc(image, object_prompt):

\"\"\"
Answers a question about the attributes of an object specified by an x,y coordinate.
Should not be used for other kinds of questions.

Args:
    image (image): Image of the scene.
    question (string): Question about the objects attribute to answer. Examples: "What color is this?", "What material is this?"
    x (int): X coordinate of the object in pixel space.
    y (int): Y coordinate of the object in pixel space. 
    

Returns:
    string: Answer to the question about the object in the image.
\"\"\"
def vqa(image, question, x, y):

\"\"\"
Returns the depth of an object specified by an x,y coordinate.

Args:
    image (image): Image of the scene.
    x (int): X coordinate of the object in pixel space.
    y (int): Y coordinate of the object in pixel space.

Returns:
    float: The depth of the object specified by the coordinates.
\"\"\"
def depth(image, x, y):

\"\"\"
Checks if two pairs of coordinates correspond to the same object.

Args:
    image (image): Image of the scene.
    x_1 (int): X coordinate of object 1 in pixel space.
    y_1 (int): Y coordinate of object 1 in pixel space.
    x_2 (int): X coordinate of object 2 in pixel space.
    y_2 (int): Y coordinate of object 2 in pixel space.

Returns:
    bool: True if object 1 is the same object as object 2, False otherwise.
\"\"\"
def same_object(image, x_1, y_1, x_2, y_2):
"""
