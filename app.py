import gradio as gr
import gradio.inputs


 #interacting with input and output directories

import pickle
from final_maybe import LanguageModel
with open('without_laplace.sav','rb') as handle:
    loaded_model = pickle.load(handle)
def fn(X_test):
    
    
    X_final_list = list(map(str, X_test.split(' ')))
    X_final=tuple(X_final_list[-2:])
    model = loaded_model
    result = model._best_candidate(X_final,0)
    
    return result
description = "Here is an interface for next word prediction using tri-gram model. Given an input, our model will predict the next word. Please make sure not to add a space after the last word."
here = gr.Interface(fn=fn,
                     inputs= gradio.inputs.Textbox( lines=1, placeholder=None, default="", label=None),
                     outputs='text',
                     title="Next Word Prediction",
                     description=description,
                     theme="default",
                     allow_flagging="auto",
                     flagging_dir='flagging records')
here.launch(inline=False,share=False)
# if __name__ == "__main__":
#     app, local_url, share_url = here.launch()


