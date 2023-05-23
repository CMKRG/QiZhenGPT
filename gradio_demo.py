import sys
import gradio as gr
import argparse
import os
import mdtex2html

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default="qizhen_model/", type=str)
parser.add_argument('--tokenizer_path',default="qizhen_model/",type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')
if args.tokenizer_path is None:
    args.tokenizer_path = args.lora_model
    if args.lora_model is None:
        args.tokenizer_path = args.base_model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

base_model = LlamaForCausalLM.from_pretrained(
    args.base_model, 
    load_in_8bit=False,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto',
    )

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenzier_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
if model_vocab_size!=tokenzier_vocab_size:
    assert tokenzier_vocab_size > model_vocab_size
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenzier_vocab_size)

model = base_model

if device==torch.device('cpu'):
    model.float()

model.eval()

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], []

def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
### Instruction:
{instruction}

### Response: """

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def predict(
    input,
    chatbot,
    history,
    max_new_tokens=512,
    top_p=0.9,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    max_memory=256,
    **kwargs,
):
    now_input = input
    chatbot.append((input, ""))
    history = history or []
    if len(history) != 0:
        input = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in history]) + \
        "### Instruction:\n" + input
        input = input[len("### Instruction:\n"):]
        if len(input) > max_memory:
            input = input[-max_memory:]
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty),
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("### Response:")[-1].strip()
    history.append((now_input, output))
    chatbot[-1] = (now_input, output)
    return chatbot, history

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">启真医学大模型</h1>""")
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 4096, value=512, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.9, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.1, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, history, max_length, top_p, temperature], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
demo.queue().launch(share=False, inbrowser=True, server_name = '0.0.0.0', server_port=16667)