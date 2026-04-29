import gradio as gr
import torch
from transformers import pipeline, AutoProcessor

model_id = "d:/Poornanand/Antigravity LLM/medgemma-4b-it-sft-lora-crc100k"

# Load the processor and pipeline
processor = AutoProcessor.from_pretrained(model_id)

# Determine device map: force to a single GPU locally to prevent multi-GPU split errors
device_map = {"": 0} if torch.cuda.is_available() else "auto"



pipe = pipeline(
    "image-text-to-text",
    model=model_id,
    processor=processor,
    torch_dtype=torch.bfloat16,
     device_map=device_map,
)

# Configuration for deterministic generation
pipe.model.generation_config.do_sample = False
pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
processor.tokenizer.padding_side = "left"

def predict_classification(image):
    question = "What is the disease?\nA: Covid19\nB: Normal\nC: Pneumonia\nD: Tuberculosis"
    return predict(image, question, max_new_tokens=10)

def predict_qa(image, question):
    if not question.strip():
        return "Please ask a question."
    return predict(image, question, max_new_tokens=150)

def predict(image, question, max_new_tokens):
    if image is None:
        return "Please upload an image first."
        
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"}
            ]
        }
    ]
    
    # Format the text with the chat template
    text_str = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Ensure the <image> token is present
    if "<image>" not in text_str:
        text_str = "<image>" + text_str
        
    # Generate the output
    outputs = pipe({"images": image, "text": text_str}, max_new_tokens=max_new_tokens)
    full_text = outputs[0]['generated_text']
    
    # Parse the output to extract the model's response
    if "model\n" in full_text:
        response = full_text.split("model\n")[-1].strip()
    else:
        response = full_text.strip()
        
    return response

# Create the Gradio interface using Blocks
with gr.Blocks() as demo:
    gr.Markdown("# MedGemma Chest X-Ray Disease Classification & QA - Poornanand")
    gr.Markdown("Upload a chest X-ray image below, then you can either automatically classify it or ask specific questions about it.")
    
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            img_input = gr.Image(type="pil", label="Upload Chest X-Ray", height=350)
        with gr.Column(scale=1):
            pass
        
    with gr.Row():
        # Column 1: Disease Classification
        with gr.Column():
            gr.Markdown("### 1. Disease Classification")
            gr.Textbox(
                value="What is the disease?\nA: Covid19\nB: Normal\nC: Pneumonia\nD: Tuberculosis",
                label="Classification Prompt (Auto-applied)",
                interactive=False,
                lines=5
            )
            btn_class = gr.Button("Classify Disease", variant="primary")
            out_class = gr.Textbox(label="Diagnosis Result")
            
        # Column 2: Custom Question & Answer
        with gr.Column():
            gr.Markdown("### 2. Ask Questions (Q&A)")
            txt_qa = gr.Textbox(lines=2, label="Ask a question about the image", placeholder="E.g., What are the visible abnormalities?")
            btn_qa = gr.Button("Ask Question", variant="primary")
            out_qa = gr.Textbox(label="Model Answer", lines=4)
            
    btn_class.click(fn=predict_classification, inputs=[img_input], outputs=[out_class])
    btn_qa.click(fn=predict_qa, inputs=[img_input, txt_qa], outputs=[out_qa])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
