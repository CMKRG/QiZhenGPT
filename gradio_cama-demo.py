from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch
import sys
import gradio as gr
import argparse
import os
import mdtex2html
from scripts.callbacks import Iteratorize, Stream
# from examples.prompter import Prompter
import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'



def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []

def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
### Instruction:
{instruction}

### Response: """


gr.Chatbot.postprocess = postprocess

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

base_model: str = "cama-path"
lora_weights: str = "lora-path"
load_8bit = False

# prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
    )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # pad
model.config.bos_token_id = tokenizer.pad_token_id = 1
model.config.eos_token_id = tokenizer.pad_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


max_memory=512

def evaluate(
    chatbot,
    instruction,
    # input=None,
    temperature=0.4,
    top_p=0.75,
    top_k=40,
    num_beams=2,
    max_new_tokens=512,
    repetition_penalty=1.3,
    stream_output=False,
    history=None,
    **kwargs,
):

    now_input = instruction
    chatbot.append((instruction, ""))
    history = history or []
    if len(history) != 0:
        instruction = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in history]) + \
        "### Instruction:\n" + instruction
        instruction = instruction[len("### Instruction:\n"):]
        if len(instruction) > max_memory:
            instruction = instruction[-max_memory:]

    prompt = generate_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )


    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # output = prompter.get_response(output)
    output = output.split("### Response:")[-1].strip()
    history.append((now_input, output))
    chatbot[-1] = (now_input, output)
    # return chatbot, history
    # yield prompter.get_response(output)
    yield chatbot, history


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">CaMA0601 - training - 3600steps </h1>""")
    current_file_path = os.path.abspath(os.path.dirname(__file__))
    # gr.Image(f'{current_file_path}/../pics/banner.png', label = 'Chinese LLaMA & Alpaca LLM')
    gr.Markdown("> 启真医学大模型")
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
            temperature = gr.components.Slider(
                minimum=0, maximum=1, value=0.4, label="Temperature"
            )
            top_p = gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            )
            top_k = gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            )
            num_beams = gr.components.Slider(
                minimum=1, maximum=4, step=1, value=2, label="Beams"
            )
            max_new_tokens = gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=512, label="Max tokens"
            )
            repetition_penalty = gr.components.Slider(
                minimum=1, maximum=2, step=0.1, value=1.3, label="Repetition Penalty"
            )
            stream_output = gr.components.Checkbox(label="Stream output")

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(evaluate, [chatbot, user_input, temperature, top_p, top_k, num_beams, max_new_tokens, repetition_penalty, stream_output, history],[chatbot, history],
                    show_progress=True)
    # submitBtn.click(predict, [user_input, chatbot, history, max_length, top_p, temperature], [chatbot, history],
    #                 show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True,
                    server_name='0.0.0.0', server_port=16666)
