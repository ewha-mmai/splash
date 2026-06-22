import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI
import time

def get_gpt_evaluator(model_name, eval_prompt):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def evaluate(prompt, correct_response, assistant_response):

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {
                    "role": "user",
                    "content": eval_prompt.format(
                        prompt=prompt,
                        correct_response=correct_response,
                        assistant_response=assistant_response
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )

        return response.choices[0].message.content

    return evaluate
    

def get_gpt5_evaluator(model_name, prompt_template, max_retries=3):
    client = OpenAI()

    def get_gpt5_evaluator(model_name, prompt_template, max_retries=3):

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def judge_fn(prompt, assistant_response, correct_response):

            full_prompt = prompt_template.format(
                prompt=prompt,
                assistant_response=assistant_response,
                correct_response=correct_response,
            )

            for attempt in range(max_retries):
                try:
                    response = client.responses.create(
                        model=model_name,
                        input=full_prompt,
                        temperature=0.0,
                    )

                    if hasattr(response, "output_text"):
                        return response.output_text.strip()

                    if response.output:
                        text = response.output[0].content[0].text
                        return text.strip()

                    return "ERROR: Empty GPT5 response"

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        return f"ERROR: {str(e)}"

        return judge_fn


def get_llama3_evaluator(
    model_path, 
    system_prompt, 
    user_prompt_template
):

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    def evaluate(prompt, correct_response, assistant_response):
        user_message = user_prompt_template.format(
            prompt=prompt,
            correct_response=correct_response,
            assistant_response=assistant_response
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        terminators = tokenizer.eos_token_id

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=terminators,
        )

        response = outputs[0][input_ids.shape[-1]:]
        decoded_output = tokenizer.decode(response, skip_special_tokens=True).strip()

        return decoded_output

    return evaluate


def get_llama3_evaluator_batch(
    model_path,
    system_prompt,
    user_prompt_template,
    max_new_tokens=120,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    def evaluate_batch(prompts, correct_responses, assistant_responses):

        messages_list = []

        for p, c, a in zip(prompts, correct_responses, assistant_responses):
            user_message = user_prompt_template.format(
                prompt=p,
                correct_response=c,
                assistant_response=a
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            messages_list.append(messages)

        inputs = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        results = []

        for i in range(outputs.size(0)):
            prompt_len = attention_mask[i].sum()
            response_tokens = outputs[i][prompt_len:]
            decoded = tokenizer.decode(
                response_tokens,
                skip_special_tokens=True
            ).strip()
            results.append(decoded)

        return results

    return evaluate_batch


def get_vicuna_evaluator(model_path, eval_prompt):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)


    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,     
        device_map="auto",
        dtype=torch.float16, 
        return_full_text=False      
    )

    def evaluate(**kwargs):
        out = pipe(
            eval_prompt.format(**kwargs),
            max_new_tokens=120,
            temperature=0.0,       
            do_sample=False,
            repetition_penalty=1.0
        )
        return "Score: " + out[0]["generated_text"]

    return evaluate


def get_vicuna_evaluator_2(
    model_path,
    eval_prompt
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16
    )

    def evaluate(**kwargs):
        prompt_text = eval_prompt.format(**kwargs)

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        if decoded.startswith(prompt_text):
            decoded = decoded[len(prompt_text):]

        if "ASSISTANT:" in decoded:
            decoded = decoded.split("ASSISTANT:")[-1]

        return decoded.strip()

    return evaluate


