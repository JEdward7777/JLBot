#https://github.com/marella/gpt4all-j
from gpt4all import GPT4All
import sys
import ctypes, re


# system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# - StableLM will refuse to participate in anything that could harm a human.
# """

system_prompt = """<|SYSTEM|># JLBot (Alpha version)
- JLBot is a helpful and harmless open-source AI language model.
- JLBot is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- JLBot is more than just an information source, JLBot is also able to write poetry, short stories, and make jokes.
- JLBot will refuse to participate in anything that could harm a human.
- JLBot was made by Joshua Lansford using GPT4All.
"""




from gpt4all.pyllmodel import LLModelPromptContext, PromptCallback, ResponseCallback, RecalculateCallback, load_llmodel_library


import random, time, asyncio
import gradio as gr

gptj = GPT4All("ggml-gpt4all-j-v1.3-groovy")


lock = asyncio.Lock()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    async def respond(message, chat_history):
        prompt = (system_prompt + 
                  "".join([f"<|USER|>{user_input}<|ASSISTANT|>{assistant_output.replace( '<|ASSISTANT|>', '' ).replace( '<|USER|>', '' )}" 
                   for (user_input,assistant_output) in chat_history ]) +
                   f"<|USER|>{message}<|ASSISTANT|>")
        
        live_result = [b""]
        def response_callback(token_id, response):
            response_decoded = response.decode('utf-8')
            print(response_decoded)
            live_result[0] += response
            #return not live_result[0].endswith( "<|")
            return not live_result[0].endswith( b"|>")
                
        gptj.model._response_callback = response_callback

        async with lock:
            result = re.sub(r'<.*?>', '', gptj.generate( prompt=prompt ))

        chat_history.append((message, result))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=8000, share=True)
