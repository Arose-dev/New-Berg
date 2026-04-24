from PIL import Image, ImageDraw
from resources.prompts.predef_api import MODULES_SIGNATURES
from resources.prompts.signature_agent import SIGNATURE_PROMPT
from resources.prompts.api_agent import API_PROMPT
from resources.prompts.program_agent import PROGRAM_PROMPT
from resources.prompts.vqa_prompts import VQA_PROMPT
from IPython.display import Markdown, display, Code, HTML
from rich.console import Console
from rich.syntax import Syntax
from rich.padding import Padding
from rich.style import Style
from openai import OpenAI
import re
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
import numpy as np
import torch
import io
from unidepth.models import UniDepthV2
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import sys
import linecache


grounding_dino = None
uni_depth = None
device = None
gpt_client = None

console = Console(highlight=False, force_terminal=False)


def initialize_modules():
    global grounding_dino
    global uni_depth
    global device
    global gpt_client

    print("Initializing OpenAI Client")
    with open("../api.key", "r") as f:
        api_key = f.read().strip()
    gpt_client = OpenAI(api_key=api_key)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Initializing GroundingDINO")
    grounding_dino = load_model(
        f"../models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        f"../models/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    print("Initializing UniDepth")
    uni_depth = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").to(device)


def remove_substring(output, substring):
    if substring in output:
        return output.replace(substring, "")
    else:
        return output


def wrap_generated_program(generated_program):
    wrapped_program = f"""
def solution_program(image):
{generated_program}
    return final_result
final_result = solution_program(image)
"""

    return wrapped_program


def correct_indentation(code_str):
    lines = code_str.split("\n")
    tabbed_lines = ["\t" + line for line in lines]
    tabbed_text = "\n".join(tabbed_lines)
    return tabbed_text


def generate(prompt, messages=None):
    if messages:
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
        )
    else:
        messages = [{"role": "user", "content": prompt}]
        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
        )
    result = response.choices[0].message.content.lstrip("\n").rstrip("\n")
    result = remove_substring(result, "```python")
    result = remove_substring(result, "```")

    return result


def load_image(im_pth):
    image = Image.open(im_pth).convert("RGB")
    image = image.resize((640, 480))
    return image


def display_predef_api():
    console.print(
        Syntax(
            MODULES_SIGNATURES,
            "python",
            theme="github",
            line_numbers=False,
            word_wrap=True,
        )
    )
    return MODULES_SIGNATURES


def display_generated_signatures(generated_signatures, generated_docstrings):
    code_to_display = ""
    for signature, docstring in zip(generated_signatures, generated_docstrings):
        code_to_display += docstring
        code_to_display += signature
    console.print(
        Syntax(
            code_to_display,
            "python",
            theme="github",
            line_numbers=False,
            word_wrap=True,
        )
    )


def display_generated_api(api):
    console.print(
        Syntax(
            api,
            "python",
            theme="github",
            line_numbers=True,
            word_wrap=True,
        )
    )


def display_generated_program(program, api):
    # Count lines in API
    api_lines = len(api.split("\n"))
    console.print(
        Syntax(
            program,
            "python",
            theme="github",
            line_numbers=True,
            start_line=api_lines + 2,
            word_wrap=True,
        )
    )


def signature_agent(predef_api, query):

    template_prompt = SIGNATURE_PROMPT
    prompt = template_prompt.format(signatures=predef_api, question=query)

    output = generate(prompt)

    docstrings = re.findall(r"<docstring>(.*?)</docstring>", output, re.DOTALL)
    signatures = re.findall(r"<signature>(.*?)</signature>", output, re.DOTALL)

    return signatures, docstrings


def api_agent(predef_signatures, gen_signatures, gen_docstrings):
    method_names = [
        re.compile(r"def (\w+)\s*\(.*\):").search(sig).group(1)
        for sig in gen_signatures
    ]
    gen_signatures_text = "".join(
        [doc + sig for doc, sig in zip(gen_signatures, gen_docstrings)]
    )

    implementations = {}
    error_count = {}
    sig_idx = 0
    while sig_idx < len(gen_signatures):
        signature = gen_signatures[sig_idx]
        docstring = gen_docstrings[sig_idx]

        if sig_idx in error_count and error_count[sig_idx] > 4:
            sig_idx += 1
            continue

        template_prompt = API_PROMPT
        prompt = template_prompt.format(
            predef_signatures=predef_signatures,
            generated_signatures=gen_signatures_text,
            docstring=docstring,
            signature=signature,
        )

        output = generate(prompt)

        implementation = re.findall(
            r"<implementation>(.*?)</implementation>", output, re.DOTALL
        )
        implementation = implementation[0]

        lines = implementation.split("\n")
        signature_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                signature_index = i
                break
        if signature_index is not None:
            implementation = "\n".join(lines[signature_index + 1 :])
        else:
            implementation = correct_indentation(implementation)

        error = False
        for i, method_name in enumerate(method_names):
            if method_name in implementation:
                error_count[sig_idx] = error_count.get(sig_idx, 0) + 1
                error = True
                break

        if not error:
            implementations[sig_idx] = implementation
            sig_idx += 1

    api = [
        gen_signatures[key] + implementations[key].strip("\n")
        for key in sorted(implementations.keys())
    ]

    merged_api = ""
    for method in api:
        merged_api += method.replace("\t", "    ")

    return merged_api


def program_agent(api, query):
    prompt = PROGRAM_PROMPT.format(
        predef_signatures=MODULES_SIGNATURES, api=api, question=query
    )
    output = generate(prompt)
    program = re.findall(r"<program>(.*?)</program>", output, re.DOTALL)
    program = correct_indentation(program[0])

    return program.replace("\t", "    ")


def transform_image(og_image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    og_image = og_image.convert("RGB")
    img = np.asarray(og_image)
    im_t, _ = transform(og_image, None)
    return img, im_t


def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        x_0, y_0, x_1, y_1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=8)

    return img1


def html_embed_image(img, size=300):
    img = img.copy()
    img.thumbnail((size, size))
    with BytesIO() as buffer:
        if img.mode == "F":
            img = img.convert("RGB")
        img.save(buffer, "jpeg")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="vertical-align:middle" src="data:image/jpeg;base64,{base64_img}">'
    )


def depth_to_grayscale(depth_map):
    # Ensure depth_map is a NumPy array of type float (if not already)
    depth_map = np.array(depth_map, dtype=np.float32)

    # Get the minimum and maximum depth values
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)

    # Avoid division by zero if the image is constant
    if d_max - d_min == 0:
        normalized = np.zeros_like(depth_map)
    else:
        normalized = (depth_map - d_min) / (d_max - d_min)

    # Scale to 0-255 and convert to unsigned 8-bit integer
    grayscale = (normalized * 255).astype(np.uint8)

    return grayscale


def dotted_image(img, points):
    # Scale dot size based on image width
    if isinstance(img, np.ndarray):
        img_width = img.shape[1]
        np_img = img.copy()
        img = Image.fromarray(np_img)
        if img.mode == "F":
            img = depth_to_grayscale(np_img)
            img = Image.fromarray(img)
            img = img.convert("RGB")
    else:
        img_width = img.size[0]

    dot_size = int(img_width * 0.02)  # 2% of image width
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for pt in points:
        x = pt[0]
        y = pt[1]

        draw.ellipse(
            (x - dot_size, y - dot_size, x + dot_size, y + dot_size),
            fill="red",
            outline="black",
        )
    return img1


def _parse_bounding_boxes(boxes, width, height):
    if len(boxes) == 0:
        return []

    bboxes = []
    for box in boxes:
        cx, cy, w, h = box
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        bboxes.append(
            [
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height),
            ]
        )
    return bboxes


def loc(image, object_prompt):
    BOX_THRESHOLD = 0.25
    TEXT_TRESHOLD = 0.25

    original_object_prompt = object_prompt
    width, height = image.size
    prompt = f"{object_prompt.replace(' ', '-')} ."
    _, img_gd = transform_image(image)

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        boxes, logits, phrases = predict(
            model=grounding_dino,
            image=img_gd,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device="cuda:0",
        )
    bboxes = _parse_bounding_boxes(boxes, width, height)

    # Generate trace HTML
    trace_html = []
    if len(bboxes) == 0:
        trace_html.append("<p>No objects found</p>")
        return [], trace_html

    trace_html.append(f"<p>Locate: {original_object_prompt}</p>")
    boxed_image = box_image(image, bboxes)
    boxed_html = html_embed_image(boxed_image)
    trace_html.append(boxed_html)

    if len(bboxes) > 1 and original_object_prompt[-1] != "s":
        original_object_prompt += "s"
    trace_html.append(f"<p>{len(bboxes)} {original_object_prompt} found</p>")
    trace_html.append(f"<p>Boxes: {bboxes}</p>")

    return bboxes, trace_html


def depth(image, bbox):
    trace_html = []
    x_mid = (bbox[0] + bbox[2]) / 2
    y_mid = (bbox[1] + bbox[3]) / 2
    with torch.no_grad():
        rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
        preds = uni_depth.infer(rgb)["depth"].squeeze().cpu().numpy()
        depth_val = preds[int(y_mid), int(x_mid)]

    trace_html.append(f"<p>Depth: ({x_mid}, {y_mid})</p>")
    dotted_im = dotted_image(preds, [[x_mid, y_mid]])
    dotted_html = html_embed_image(dotted_im)
    trace_html.append(dotted_html)
    trace_html.append(f"<p>Depth: {depth_val}</p>")

    return depth_val, trace_html


def _vqa_predict(img, question, holistic=False):
    prompt = VQA_PROMPT.format(question=question)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }
    ]
    output = generate("", messages=messages)
    answer = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].lower()
    return answer


def vqa(image, question, bbox):
    trace_html = []
    if bbox is None:
        answer = _vqa_predict(image, question, holistic=True)
        boxed_image = image
    else:
        boxed_image = box_image(image, [bbox])
        answer = _vqa_predict(boxed_image, question)

    trace_html.append(f"<p>Question: {question}</p>")
    trace_html.append(html_embed_image(boxed_image, 300))
    trace_html.append(f"<p>Answer: {answer}</p>")

    return answer.lower(), trace_html


def _get_iou(box1, box2):
    # Coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Width and height of the intersection rectangle
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)

    # Area of the intersection
    area_inter = width_inter * height_inter

    # Area of both bounding boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union of both bounding boxes
    area_union = area_box1 + area_box2 - area_inter

    # IoU calculation
    iou = area_inter / area_union if area_union != 0 else 0

    return iou


def same_object(image, bbox1, bbox2):
    answer = _get_iou(bbox1, bbox2) > 0.92
    trace_html = []
    boxed_image = box_image(image, [bbox1, bbox2])
    trace_html.append(f"<p>Same Object?</p>")
    trace_html.append(html_embed_image(boxed_image, 300))
    trace_html.append(
        f"<p>IoU: {_get_iou(bbox1, bbox2):.3f}, Same object: {answer}</p>"
    )
    return answer, trace_html


def get_2D_object_size(image, bbox):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])

    trace_html = []
    trace_html.append(f"<p>2D Object Size</p>")
    boxed_image = box_image(image, [bbox])
    trace_html.append(html_embed_image(boxed_image, 300))
    trace_html.append(f"<p>Width: {width}, Height: {height}</p>")

    return (width, height), trace_html


def execute_program(program, image, api):
    wrapped_program = wrap_generated_program(program)

    executable_program = api + wrapped_program

    # Store program lines in a list for reference
    program_lines = executable_program.split("\n")

    # Create a function to get line text from our stored program
    def get_line(line_no):
        # Adjust line number to 0-based index
        idx = line_no - 1
        if 0 <= idx < len(program_lines):
            return program_lines[idx]
        return ""

    # parse API methods
    api_methods = re.findall(r"def (\w+)\s*\(.*\):", api)

    # Create a trace string to record execution
    html_trace = []

    # Create namespace for execution
    def _traced_loc(*args):
        result, html = loc(*args)
        html_trace.extend(html)
        return result

    def _traced_vqa(*args):
        result, html = vqa(*args)
        html_trace.extend(html)
        return result

    def _traced_depth(*args):
        result, html = depth(*args)
        html_trace.extend(html)
        return result

    def _traced_get_2D_object_size(*args):
        result, html = get_2D_object_size(*args)
        html_trace.extend(html)
        return result

    def _traced_same_object(*args):
        result, html = same_object(*args)
        html_trace.extend(html)
        return result

    # Create a custom trace function to track line execution
    def trace_lines(frame, event, arg):
        if event == "line":
            method_name = frame.f_code.co_name
            if method_name == "solution_program" or method_name in api_methods:
                line_no = frame.f_lineno
                line = get_line(line_no).strip()
                if len(line) > 0:
                    html_trace.append(
                        f"<p><code>[{method_name}] Line {line_no}: {line}</code></p>"
                    )
        return trace_lines

    namespace = {
        "loc": _traced_loc,
        "vqa": _traced_vqa,
        "depth": _traced_depth,
        "image": image,
        "get_2D_object_size": _traced_get_2D_object_size,
        "same_object": _traced_same_object,
    }

    # Set up the trace function
    sys.settrace(trace_lines)

    try:
        # Execute the program
        exec(executable_program, namespace)

    finally:
        # Disable tracing
        sys.settrace(None)

    final_result = namespace["final_result"]

    # Return both the text trace and HTML trace
    return final_result, "\n".join(html_trace)


def display_result(final_result, image, question, ground_truth):
    result_html = []
    result_html.append(f"<p>Question: {question}</p>")
    result_html.append(html_embed_image(image, 300))
    result_html.append(f"<p>Result: {final_result}</p>")
    result_html.append(f"<p>Ground Truth: {ground_truth}</p>")
    return "\n".join(result_html)
