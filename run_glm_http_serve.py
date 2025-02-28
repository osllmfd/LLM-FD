import os
import platform
import random
import re
import string
import time
import uuid

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Detect operating system
is_win = (platform.system() == 'Windows')
if is_win:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure device and model paths
device = "cuda" if torch.cuda.is_available() else "cpu"
file = 'glm-4-9b-chat'
model_path = f'./models/glm/{file}'

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
    trust_remote_code=True
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def generate_id(prefix: str, k=29) -> str:
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return f"{prefix}{suffix}"


@app.route('/exct', methods=['POST'])
def exct():
    # Get request body as raw text
    raw_data = request.get_data(as_text=True)

    # Use regex to match all text after "分析：" or "判断：" with DOTALL flag
    match = re.search(r'(分析：|判断：)(.*)', raw_data, re.DOTALL)
    if match:
        # Extract matched text
        extracted_text = match.group(2)

        # Remove "快速" (fast) term
        without_fast = re.sub(r'快速', '', extracted_text)
        without_fast = re.sub(r'幅度为', '了', without_fast)

        # Standardize terminology
        without_fast = re.sub(r'超过', r'大于', without_fast)
        without_fast = re.sub(r'上升', r'增大', without_fast)
        without_fast = re.sub(r'下降', r'减小', without_fast)

        # Remove all whitespace
        cleaned_text = re.sub(r'\s+', '', without_fast)
        return cleaned_text
    else:
        # Process entire raw data if no match found
        without_fast = re.sub(r'快速', '', raw_data)
        without_fast = re.sub(r'幅度为', '了', without_fast)
        without_fast = re.sub(r'超过', r'大于', without_fast)
        without_fast = re.sub(r'上升', r'增大', without_fast)
        without_fast = re.sub(r'下降', r'减小', without_fast)
        cleaned_text = re.sub(r'\s+', '', without_fast)
        return cleaned_text


# Define API endpoint
@app.route('/chat/completions', methods=['POST'])
def completions():
    data = request.json

    print(f'Chat input data: {data=}')

    # Validate request data
    if 'messages' not in data:
        return jsonify({'error': 'Missing messages field'}), 400

    messages = data['messages']
    temperature_ = data.get('temperature', 0.8)

    # Text generation logic
    inputs = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )
    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 128000, "do_sample": True, "top_k": 1, 'temperature': temperature_}

    outputs = model.generate(**inputs, ​**gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Return in OpenAI API format
    response = {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": file,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length" if len(response_text) >= gen_kwargs['max_length'] else "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(inputs['input_ids'][0]),
            "completion_tokens": len(outputs[0]),
            "total_tokens": len(inputs['input_ids'][0]) + len(outputs[0])
        }
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=53121)
