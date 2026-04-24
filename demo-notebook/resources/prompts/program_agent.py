PROGRAM_PROMPT = """
You are an expert logician capable of answering spatial reasoning problems with code. You excel at using a predefined API to break down a difficult question into simpler parts to write a program that answers spatial and complex reasoning problem.

Answer the following question using a program that utilizes the API to decompose more complicated tasks and solve the problem. 

I am going to give you two examples of how you might approach a problem in psuedocode, then I will give you an API and some instructions for you to answer in real code.

Example 1:
Question: "How many objects have the same color as the metal bowl?"
Solution:
1) Set a counter to 0
2) Find all the bowls (loc(image, 'bowls')).
3) If bowls are found, loop through each of the bowls found.
4) For each bowl found, check if the material of this bowl is metal. Store the metal bowl if you find it and break from the loop.
5) Find and store the color of the metal bowl.
6) Find all the objects.
7) For each object O, check if O is the same object as the small bowl (same_object(image, metal_bowl_bbox, object_bbox)). If it is, skip it.
8) For each O you don't skip, check if the color of O is the same as the color of the metal bowl.
9) If it is, increment the counter.
10) When you are done looping, return the counter.

Example 2:
Question: "How many objects of the same height as the mug would you have to stack to achieve an object the same height as the cabinet?"
Solution:
1) Locate the mug (loc(image, "mug"))
2) Locate the cabinet (loc(image, "cabinet"))
3) Find the 2D height of the mug, multiply it by the depth and store this value.
4) Find the 2D height of the cabinet, multiply it by the depth and store this value.
5) Return the height of the cabinet divided by the height of the mug, do NOT round.

Example 3:
Question: "How many mugs are there in the dishwasher?"
Solution:
1) Locate all the mugs (loc(image, "mug"))
2) Initialize a counter to 0
3) For each mug you found, ask VQA if the mug is in the dishwasher (vqa(image, "Is this mug in the dishwasher?", mug_bbox))
4) If the output is "yes" then increment the counter.
5) Return the counter.

Example 4:
Question: "How many plates are on the table?"
Solution:
1) Locate all the plates (loc(image, "plate"))
2) Return the number of plates located.

Now here is an API of methods, you will want to solve the problem in a logical and sequential manner as I showed you

------------------ API ------------------
{predef_signatures}
{api}
------------------ API ------------------

Please do not use synonyms, even if they are present in the question.
Using the provided API, output a program inside the tags <program></program> to answer the question. 
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

Here are some helpful definitions:
1) 2D distance/size refers to distance/size in pixel space.
2) 3D distance/size refers to distance/size in the real world. 3D size is equal to 2D size times the depth of the object.
3) We define (width, height, length) as the values along the (x, y, z) axis. Width = x axis, height = y axis, length = z axis.
4) "Depth" measures distance from the camera in 3D.

Here are some helpful tips: 
1) When you need to search over objects satisfying a condition, remember to check all the objects that satisfy the condition and don't just return the first one. 
2) You already have an initialized variable named "image" - no need to initialize it yourself! 
3) When searching for objects to compare to a reference object, make sure to remove the reference object from the retrieved objects. You can check if two objects are the same with the same_object method.
4) If two objects have the same 2D width, then the object with the largest depth has the largest 3D width.
5) If two objects have the same 2D height, then the object with the largest depth has the largest 3D height.
6) 2D sizes convey the height and width in IMAGE SPACE. To convert to height and width in 3D space, it needs to be multiplied by the depth!
7) Some questions may present hypothesis measurements (e.g. "X is 3.5m wide"), these are hypothesis measurements and should be used ONLY to scale your outputs accordingly.
8) Do NOT round your answers! Always leave your answers as decimals even when it feels intuitive to round or ceiling your answer - do not do it!
9) When a query asks to find all objects in a container just count the number of objects.

Again, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
It is critical that the final answer is stored in a variable called "final_result".
Ensure that the answer is either yes/no, one word, or one number.

AGAIN, answer the question by using the provided API to write a program in the tags <program></program> and ensure the program stores the answer in a variable called "final_result".
You do not need to define a function to answer the question - just write your program in the tags. Assume "image" has already been initialized - do not modify it!
<question>{question}</question>
"""
