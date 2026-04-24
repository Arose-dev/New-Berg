API_PROMPT = """
You are an expert at implementing methods accoring to a given docstring and signature.
Implement a method given a docstring and method signature, using the API as necessary.

API:
{predef_signatures}

{generated_signatures}

Here are some examples of how to implement a method given its docstring and signature:

<docstring>
\"\"\"
Gets the material of the given object.

Args:
    image (IMAGE): Image that the object is contained in.
    bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the object.

Returns:
    str: Material of the object.
\"\"\"
</docstring>
<signature>def object_material(image, bbox):</signature>
<implementation>
material = vqa(image=image, question='What material is this object?', bbox=bbox)
return material
</implementation>

<docstring>
\"\"\"
Checks if an object 1 is in front of object 2.

Args:
    image (IMAGE): Image that the object is contained in.
    bbox1 (list): A bounding box [xmin, ymin, xmax, ymax] containing object1.
    bbox2 (list): A bounding box [xmin, ymin, xmax, ymax] containing object2.

Returns:
    bool: True if object 1 is in front of object 2, False otherwise
\"\"\"
</docstring>
<signature>def in_front_of(image, bbox1, bbox2):</signature>
<implementation>
depth_1 = depth(image, bbox1)
depth_2 = depth(image, bbox2)
return depth_1 < depth_2
</implementation>

<docstring>
\"\"\"
Calculates the ratio of height between a base and target object.

Args:
    image (IMAGE): Image to search for objects in
    base_bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the base object.
    target_bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing target object.

Returns:
    float: base object 3D height divided by target object 3D height.
\"\"\"
</docstring>
<signature>def _calculate_height_ratio(image, base_bbox, target_bbox):</signature>
<implementation>
base_width, base_height = get_2D_object_size(image, base_bbox)
base_depth = depth(image, base_bbox)
base_3D_height = base_height * base_depth

target_width, target_height = get_2D_object_size(image, target_bbox)
target_depth = depth(image, target_bbox)
target_3D_height = target_height * target_depth

return target_3D_height / base_3D_height
</implementation>

<docstring>
\"\"\"
Checks if a target object and a base object share a spatial relationship (e.g. "next to", "on top of", "on")

Args:
    image (IMAGE): Image that the object is contained in.
    base_object (str): Description of base object.
    relation (str): Relation to evaluate (e.g. "next to", "on top of", "on")
    target_bbox (list): A bounding box [xmin, ymin, xmax, ymax] containing the target object.

Returns:
    bool: True if the target object and the base object share the relation.
\"\"\"
</docstring>
<signature>def evaluate_object_relation(image, base_object, relation, target_bbox):</signature>
<implementation>
query = 'Is this object ' + relation + 'to ' + base_object + '?'
return vqa(image=image, question=query, target_bbox)
</implementation>


Here are some helpful definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) We define (width, height, length) as the values along the (x, y, z) axis. Width = x axis, height = y axis, length = z axis.
4) "Depth" measures distance from the camera in 3D.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have an initialized variable named "image" - no need to initialize it yourself.
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) Do not assume that the objects you see in these questions are all of the objects you will see, keep the methods general.
5) If two objects have the same 2D width, then the object with the largest depth has the largest 3D width.
6) If two objects have the same 2D height, then the object with the largest depth has the largest 3D height.
7) 2D sizes convey the height and width in IMAGE SPACE. To convert to height and width in 3D space, it needs to be multiplied by the depth!
8) Some questions may present hypothesis measurements (e.g. "X is 3.5m wide"), these are hypothesis measurements and should be used ONLY to scale your outputs accordingly.
9) Do NOT round your answers! Always leave your answers as decimals. If the question asks "how many X do you need to get to Y" you should NOT round - leave your answer as a floating point division.
10) To determine if an object is in another object - use VQA. For example to determine if a book is in the shelf vqa(image, 'Is the book in the shelf?', book_bbox)


Do not define new methods here, simply solve the problem using the existing methods.

Now, given the following docstring and signature, implement the method, using the API specification as necessary. Output the implementation inside <implementation></implementation>.

Again, Output the implementation inside <implementation></implementation>.

<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>
"""