import numpy as np
from openai import OpenAI
import json
import os
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import time
import torch


DEFAULT_TEXT_MODEL = os.getenv(
    "PARASAIL_DEPLOYMENT_NAME",
    os.getenv("MODEL_NAME", "Qwen2.5-72B-Instruct"),
)


def correct_indentation(code_str):
    lines = code_str.split("\n")
    tabbed_lines = ["\t" + line for line in lines]
    tabbed_text = "\n".join(tabbed_lines)
    return tabbed_text


def replace_tabs_with_spaces(code_str):
    return code_str.replace("\t", "    ")


def untab(text):
    lines = text.split("\n")
    untabbed_lines = []
    for line in lines:
        if line.startswith("\t"):
            untabbed_lines.append(line[1:])
        elif line.startswith("    "):
            untabbed_lines.append(line[4:])
        else:
            untabbed_lines.append(line)
    untabbed_text = "\n".join(untabbed_lines)
    return untabbed_text


def get_methods_from_json(api_json):
    methods = []
    namespace = {}
    for method_info in api_json:

        signature = method_info["signature"]
        if "def" in method_info["implementation"]:
            full_method = method_info["implementation"]
        else:
            full_method = signature + "\n" + method_info["implementation"]
        methods.append(full_method)

    return methods, namespace


class Generator:
    def __init__(self, model_name=DEFAULT_TEXT_MODEL, temperature=0.7, api_key_path="./api.key"):
        self.temperature = temperature
        self.model_name = model_name
        self.api_key_path = api_key_path
        # Supports any OpenAI-compatible API (Parasail, Fireworks, OpenAI, etc.)
        # Set these in a .env file or as environment variables:
        # - API_KEY: your API key (falls back to PARASAIL_API_KEY, then ./api.key)
        # - API_BASE_URL: your provider's base URL (falls back to PARASAIL_BASE_URL, OPENAI_BASE_URL)
        # - MODEL_NAME: the model to use (set at module level above)
        api_key = os.getenv("API_KEY") or os.getenv("PARASAIL_API_KEY")
        if not api_key:
            with open(self.api_key_path, "r") as file:
                api_key = file.read().strip()

        base_url = os.getenv("API_BASE_URL") or os.getenv("PARASAIL_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not base_url and os.getenv("PARASAIL_API_KEY"):
            base_url = "https://api.parasail.io/v1"

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def _log(self, message):
        print(f"[Generator:{self.model_name}] {message}", flush=True)

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def generate(self, prompt, messages=None):
        new_messages = None
        if self.model_name == "gpt-3.5-turbo-instruct":
            self._log("starting completion request")
            response = self.client.Completion.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=1024,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                logprobs=11,
            )

            result = response.choices[0]["text"].lstrip("\n").rstrip("\n")
            result = self.remove_substring(result, "```python")
            result = self.remove_substring(result, "```")

        else:
            if not messages:
                messages = [{"role": "user", "content": prompt}]
            self._log(f"starting chat request with {len(messages)} message(s)")
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
            except Exception as e:
                self._log(f"request failed: {type(e).__name__}: {e}")
                message = str(e).lower()
                if any(
                    marker in message
                    for marker in (
                        "403",
                        "permissiondeniederror",
                        "doesn't exist",
                        "isn't accessible",
                        "invalid_request_error",
                        "401",
                    )
                ):
                    self._log("failing fast because this looks like a credentials/model-access issue")
                    raise
                self._log("sleeping 60s before retry")
                time.sleep(60)
                return self.generate(prompt, messages)
            new_messages = messages
            result = response.choices[0].message.content.lstrip("\n").rstrip("\n")
            self._log("request completed")
            result = self.remove_substring(result, "```python")
            result = self.remove_substring(result, "```")
            new_messages.append(
                {
                    "role": response.choices[0].message.role,
                    "content": result,
                }
            )
        return result, new_messages


def docstring_from_json(json_file):
    with open(json_file, "r") as file:
        api_data = json.load(file)

    docstring = ""
    for module in api_data.get("modules", []):
        docstring += f'"""\n'
        docstring += f"{module['description']}\n\n"
        if module["arguments"]:
            docstring += "Args:\n"
            for arg in module["arguments"]:
                docstring += (
                    f"    {arg['name']} ({arg['type']}): {arg['description']}\n"
                )
        if "returns" in module:
            docstring += f"\nReturns:\n"
            docstring += f"    {module['returns']['type']}: {module['returns']['description']}\n\"\"\""
        docstring += f"\n{module['name']}("
        args = [arg["name"] for arg in module["arguments"]]
        docstring += ", ".join(args) + ")\n\n"

    return docstring.strip()

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


def box_image(img, boxes):
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for box in boxes:
        x_0, y_0, x_1, y_1 = box[0], box[1], box[2], box[3]
        draw.rectangle([x_0, y_0, x_1, y_1], outline="red", width=8)

    return img1


def dotted_image(img, points):
    # Scale dot size based on image width
    if isinstance(img, np.ndarray):
        img_width = img.shape[1]
        np_img = img.copy()
        img = Image.fromarray(np_img)
        if img.mode == 'F':
            img = depth_to_grayscale(np_img)
            img = Image.fromarray(img)
            img = img.convert('RGB')
    else:
        img_width = img.size[0]

    
    dot_size = int(img_width * 0.02) # 2% of image width
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


def html_embed_image(img, size=300):
    img = img.copy()
    img.thumbnail((size, size))
    with BytesIO() as buffer:
        if img.mode == 'F':
            img = img.convert('RGB')
        img.save(buffer, "jpeg")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
    return (
        f'<img style="vertical-align:middle" src="data:image/jpeg;base64,{base64_img}">'
    )
    

class TimeoutException(Exception):
    pass


def set_devices():
    # select the device for computation — CUDA (NVIDIA GPU) is strongly preferred
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: {device} (Apple Silicon — NOT recommended, use CUDA for best results)")
    else:
        device = torch.device("cpu")
        print(f"Device: {device} (WARNING: No GPU detected, inference will be very slow)")

    if device.type == "cuda":
        gpu_capability = torch.cuda.get_device_properties(0).major
        if gpu_capability >= 8:
            # Ampere+ GPUs support bfloat16 and tf32
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            # Older GPUs (Turing, Volta, etc.) use float16
            torch.autocast("cuda", dtype=torch.float16).__enter__()
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device


def timeout_handler(signum, frame):
    raise TimeoutException(
        "The script took too long to run. There is likely a Recursion Error. Ensure that you are not calling a method with infinite recursion."
    )


def remove_python_text(output):
    substring = "```python"
    if substring in output:
        output = output.replace(substring, "")

    substring = "```"
    if substring in output:
        output = output.replace(substring, "")

    return output
